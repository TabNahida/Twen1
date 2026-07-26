from __future__ import annotations

import signal
from pathlib import Path

import pytest

from twen.runtime.signals import ImmediateExit, SignalController


def test_sigint_requests_graceful_checkpoint_then_second_sigint_exits() -> None:
    controller = SignalController()
    controller.handle_signal(signal.SIGINT)
    decision = controller.poll()
    assert decision.should_checkpoint
    assert decision.should_stop
    assert not decision.continue_after_checkpoint
    assert decision.reason == "sigint"

    with pytest.raises(ImmediateExit) as raised:
        controller.handle_signal(signal.SIGINT)
    assert raised.value.exit_code == 130
    assert controller.hard_stop_requested


def test_sigterm_requests_checkpoint_and_exit() -> None:
    controller = SignalController()
    controller.handle_signal(signal.SIGTERM)
    decision = controller.poll()
    assert decision.should_checkpoint
    assert decision.should_stop
    assert decision.reason == "sigterm"


def test_handler_can_reenter_controller_lock_on_main_thread() -> None:
    controller = SignalController()
    with controller._lock:  # Simulate a signal interrupting poll between bytecodes.
        controller.handle_signal(signal.SIGTERM)
    assert controller.poll().should_stop


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is POSIX-only")
def test_sigusr1_checkpoints_and_continues_until_acknowledged() -> None:
    controller = SignalController()
    controller.handle_signal(signal.SIGUSR1)
    decision = controller.poll()
    assert decision.should_checkpoint
    assert not decision.should_stop
    assert decision.continue_after_checkpoint

    controller.acknowledge_checkpoint(decision.checkpoint_generation)
    assert not controller.poll().should_checkpoint


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is POSIX-only")
def test_ack_does_not_erase_a_newer_checkpoint_request() -> None:
    controller = SignalController()
    controller.handle_signal(signal.SIGUSR1)
    first_generation = controller.poll().checkpoint_generation
    controller.handle_signal(signal.SIGUSR1)
    controller.acknowledge_checkpoint(first_generation)
    assert controller.poll().should_checkpoint


def test_stop_file_is_consumed_and_requests_graceful_stop(tmp_path: Path) -> None:
    stop_file = tmp_path / "STOP"
    stop_file.write_text("please stop\n")
    controller = SignalController(stop_file)
    decision = controller.poll()
    assert decision.should_checkpoint
    assert decision.should_stop
    assert decision.reason == "stop-file"
    assert not stop_file.exists()


def test_install_and_restore_preserve_previous_handlers() -> None:
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    controller = SignalController()
    try:
        controller.install()
        assert signal.getsignal(signal.SIGINT) == controller.handle_signal
        assert signal.getsignal(signal.SIGTERM) == controller.handle_signal
    finally:
        controller.restore()
    assert signal.getsignal(signal.SIGINT) == previous_int
    assert signal.getsignal(signal.SIGTERM) == previous_term
