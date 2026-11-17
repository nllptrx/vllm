# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("hidden_size", [256, 2048])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_gemma_rms_norm_batch_invariant(
    hidden_size, with_residual, compiled, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(19)
    with set_current_vllm_config(VllmConfig()):
        norm = GemmaRMSNorm(hidden_size).to(device="cuda", dtype=torch.bfloat16)
        norm.weight.normal_(0, 0.1)
        x = torch.randn(9, 16, hidden_size, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x) if with_residual else None
        forward = norm.forward_native
        if compiled:
            forward = torch.compile(forward, fullgraph=True, dynamic=True)
        single = forward(x[:1], residual[:1] if residual is not None else None)
        batch = forward(x, residual)
        if with_residual:
            assert residual is not None
            single_output, single_residual = single
            output, residual_output = batch
            assert torch.equal(single_residual, residual_output[:1])
            expected_residual = (x.float() + residual.float()).to(x.dtype)
            assert torch.equal(residual_output, expected_residual)
        else:
            single_output, output = single, batch
        assert torch.isfinite(output).all()
        assert torch.equal(single_output, output[:1])
        summed = x.float() if residual is None else x.float() + residual.float()
        expected = summed * torch.rsqrt(
            summed.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon
        )
        expected = (expected * (norm.weight.float() + 1)).to(x.dtype)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)


@torch.inference_mode()
def test_gemma_rms_norm_preserves_residual_precision(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    with set_current_vllm_config(VllmConfig()):
        norm = GemmaRMSNorm(256).to(device="cuda", dtype=torch.bfloat16)
        norm.weight.fill_(0.095703125)
        x = torch.ones(1, 256, device="cuda", dtype=torch.bfloat16)
        residual = torch.zeros_like(x)
        residual[:, :128] = 0.00390625
        output, residual_output = norm.forward_native(x, residual)
        summed = x.float() + residual.float()
        expected = summed * torch.rsqrt(
            summed.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon
        )
        expected = (expected * (norm.weight.float() + 1)).to(x.dtype)
        assert torch.equal(output, expected)
        assert torch.equal(residual_output, summed.to(x.dtype))
        assert not torch.equal(output[:, :128], output[:, 128:])
