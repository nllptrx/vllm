# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch

from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize(
    "partitions",
    [
        [1024, 1152],
        [1056, 1120],
        [1088, 1088],
        [2112, 64],
        [1056, 1056, 64],
        [1024, 1, 1151],
        [23] + [1] * 74,
    ],
)
@torch.inference_mode()
def test_gdn_batch_invariant_partitions(partitions, monkeypatch):
    monkeypatch.setattr(qwen_gdn_linear_attn.envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_k_heads, num_v_heads, head_dim = 2, 4, 128
    conv_dim = (2 * num_k_heads + num_v_heads) * head_dim
    num_tokens = sum(partitions)
    x = torch.randn(num_tokens, conv_dim, device=device, dtype=dtype)
    a = torch.randn(num_tokens, num_v_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    weight = torch.randn(conv_dim, 1, 4, device=device, dtype=dtype) * 0.1
    a_log = torch.randn(num_v_heads, device=device)
    dt_bias = torch.randn_like(a_log)
    conv_shape, state_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, num_k_heads, num_v_heads, head_dim, head_dim, 4, num_spec=0
    )

    def run(sizes):
        layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
        torch.nn.Module.__init__(layer)
        layer.prefix = "probe"
        layer.enable_packed_recurrent_decode = True
        layer.tp_size = 1
        layer.num_k_heads = num_k_heads
        layer.num_v_heads = num_v_heads
        layer.head_k_dim = layer.head_v_dim = head_dim
        layer.key_dim = num_k_heads * head_dim
        layer.value_dim = num_v_heads * head_dim
        layer.activation = "silu"
        layer.A_log = a_log
        layer.dt_bias = dt_bias
        layer.conv1d = types.SimpleNamespace(weight=weight, bias=None)
        layer.kv_cache = (
            torch.zeros(2, *conv_shape, device=device, dtype=dtype),
            torch.zeros(2, *state_shape, device=device, dtype=torch.float32),
        )
        layer.chunk_gated_delta_rule = chunk_gated_delta_rule
        layer._bi_decode_cu = None
        start = 0
        outputs = []
        for count in sizes:
            cu = torch.tensor([0, count], device=device, dtype=torch.int32)
            indices = torch.ones(1, device=device, dtype=torch.int32)
            initial = torch.tensor([start > 0], device=device)
            is_prefill = count > 1
            metadata = GDNAttentionMetadata(
                num_prefills=int(is_prefill),
                num_prefill_tokens=count if is_prefill else 0,
                num_decodes=int(not is_prefill),
                num_decode_tokens=int(not is_prefill),
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=count,
                has_initial_state=initial if is_prefill else None,
                non_spec_query_start_loc=cu,
                non_spec_query_start_loc_cpu=[0, count],
                non_spec_state_indices_tensor=indices,
                prefill_query_start_loc=cu,
                prefill_query_start_loc_cpu=[0, count],
                prefill_state_indices=indices,
                prefill_has_initial_state=initial,
            )
            context = types.SimpleNamespace(attn_metadata={layer.prefix: metadata})
            monkeypatch.setattr(
                qwen_gdn_linear_attn,
                "get_forward_context",
                lambda context=context: context,
            )
            output = torch.zeros(
                count, num_v_heads, head_dim, device=device, dtype=dtype
            )
            span = slice(start, start + count)
            layer._forward_core(
                x[span].clone(), b[span].clone(), a[span].clone(), output
            )
            outputs.append(output)
            # Reload persistent state as a prefix-cache copy would.
            layer.kv_cache = tuple(t.clone() for t in layer.kv_cache)
            start += count
        return torch.cat(outputs), *layer.kv_cache

    expected = run([num_tokens])
    actual = run(partitions)
    for result, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
