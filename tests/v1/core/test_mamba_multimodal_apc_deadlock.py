# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.sched.scheduler import Scheduler

from .utils import create_requests

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("gap", [1, 7, 15])
def test_mamba_alignment_never_collapses_positive_multimodal_gap(gap: int):
    """A sub-block gap before the next image must make forward progress.

    With APC enabled, encoder-cache contention can schedule only the text gap
    between two adjacent multimodal inputs. The old block alignment rounded a
    positive gap smaller than the Mamba block size to zero, leaving the request
    permanently running while producing no tokens.
    """
    block_size = 16
    request = create_requests(
        num_requests=1,
        num_tokens=block_size * 4,
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=block_size * 2),
                PlaceholderRange(offset=block_size * 2 + gap, length=block_size),
            ]
        ],
        block_size=block_size,
    )[0]
    request.num_computed_tokens = block_size * 2
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle=False,
    )

    scheduled = Scheduler._mamba_block_aligned_split(
        scheduler,
        request,
        num_new_tokens=gap,
    )

    assert scheduled == gap
    assert scheduled > 0


def test_mamba_alignment_still_rounds_full_chunks_to_blocks():
    block_size = 16
    request = create_requests(
        num_requests=1,
        num_tokens=block_size * 8,
        block_size=block_size,
    )[0]
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle=False,
    )

    scheduled = Scheduler._mamba_block_aligned_split(
        scheduler,
        request,
        num_new_tokens=block_size * 2 + 7,
    )

    assert scheduled == block_size * 2
