"""Static lifecycle contracts for production-only CUDA training references."""

from __future__ import annotations

import inspect

from twen.training.engine import run_training


def test_forward_hidden_tuples_are_released_before_interrupt_or_next_microbatch() -> None:
    """Guard the explicit release that a tiny CPU model cannot measure meaningfully.

    ``run_training`` is the only production optimizer loop and requires the
    full CUDA model/data stack.  Inspecting its narrow post-backward window
    keeps this regression test CPU-only while ensuring the 1+ GiB teacher
    tuple and the student hidden-state tuple cannot survive into checkpoint
    coordination or the next microbatch's forward RHS.
    """

    source = inspect.getsource(run_training)
    backward = source.index("scaled_loss.backward()")
    metrics_accumulated = source.index("for key, value in values.items()", backward)
    student_release = source.index("del outputs", backward)
    teacher_release = source.index("del teacher_outputs", backward)
    next_control = source.index("_coordinate_control(", backward)
    post_backward = source[backward:next_control]

    assert backward < metrics_accumulated < student_release < next_control
    assert backward < metrics_accumulated < teacher_release < next_control
    assert "del anchor_hidden_states" in post_backward
    assert "del scaled_loss, loss, target_mean" in post_backward
    assert "del batch, host_batch" in post_backward
