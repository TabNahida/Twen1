from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path
from typing import Any

import pytest

import twen.runtime.checkpoint as checkpoint_module
from twen.runtime.checkpoint import (
    CheckpointCorruptError,
    CheckpointManager,
    IncompatibleCheckpointError,
    RunLock,
    RunLockedError,
    stable_fingerprint,
)
from twen.runtime.state import (
    DataCursor,
    RNGState,
    StateVersionError,
    TrainerState,
    capture_committed_boundary,
)


class PickleBackend:
    """Small CPU-only backend; production state backends are tested by PyTorch."""

    name = "test-pickle"

    def save(self, stateful: dict[str, Any], path: Path) -> None:
        path.mkdir(parents=True)
        with (path / "state.pkl").open("wb") as handle:
            pickle.dump(dict(stateful), handle)

    def load(self, stateful: dict[str, Any] | None, path: Path) -> dict[str, Any]:
        with (path / "state.pkl").open("rb") as handle:
            saved = pickle.load(handle)
        if stateful is None:
            return saved
        stateful.clear()
        stateful.update(saved)
        return stateful


def _state(step: int = 0, *, micro_step: int = 0) -> TrainerState:
    return TrainerState(
        run_id="unit-test",
        stage="dense-oracle",
        global_step=step,
        committed_tokens=step * 4096,
        micro_step_in_accumulation=micro_step,
        gradient_accumulation_steps=4,
        global_batch_tokens=4096,
        micro_batch_tokens_per_rank=1024,
        world_size=1,
        top_k=8,
        loss_weights={"ntp": 1.0},
    )


def _cursor(step: int = 0) -> DataCursor:
    return DataCursor(
        shard_index=step // 2,
        sample_index=step * 10,
        global_sample_index=step * 10,
        global_token_index=step * 4096,
        shuffle_seed=123,
    )


def _save(manager: CheckpointManager, step: int, kind: str = "periodic") -> Path:
    return manager.save(
        {"model": {"weight": step}, "optimizer": {"step": step}},
        trainer_state=_state(step),
        data_cursor=_cursor(step),
        rng_state=RNGState.capture(),
        critical_fingerprint={"lr": 1e-4, "experts": 8},
        data_fingerprint={"manifest": "abc"},
        kind=kind,  # type: ignore[arg-type]
    )


def test_versioned_state_round_trip_and_future_version_rejected() -> None:
    state = _state(7)
    cursor = _cursor(7)
    assert TrainerState.from_dict(state.to_dict()) == state
    assert DataCursor.from_dict(cursor.to_dict()) == cursor

    newer = state.to_dict()
    newer["version"] = 999
    with pytest.raises(StateVersionError):
        TrainerState.from_dict(newer)


def test_runtime_cursor_wraps_deterministic_global_cursor_state() -> None:
    global_state = {
        "schema_version": 1,
        "dataset_fingerprint": "a" * 64,
        "dataset_size": 100,
        "seed": 3407,
        "next_global_sample": 23,
        "committed_tokens": 4096,
        "shuffle": True,
    }
    cursor = DataCursor.from_global_cursor_state(global_state)
    assert cursor.global_sample_index == 23
    assert cursor.global_token_index == 4096
    assert cursor.to_global_cursor_state() == global_state


def test_checkpoint_refuses_mismatched_committed_token_counters(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    cursor = _cursor(1)
    cursor.global_token_index -= 1
    with pytest.raises(ValueError, match="committed_tokens disagrees"):
        manager.save(
            {"model": {"weight": 1}},
            trainer_state=_state(1),
            data_cursor=cursor,
            critical_fingerprint="critical",
            data_fingerprint="data",
        )


def test_rng_capture_restore_replays_python_stream() -> None:
    random.seed(2026)
    snapshot = RNGState.capture()
    expected = [random.random() for _ in range(3)]
    snapshot.restore()
    assert [random.random() for _ in range(3)] == expected


def test_new_world_size_rank_gets_deterministic_independent_rng_stream() -> None:
    source = RNGState.capture()
    rank_two_first = source.fork_for_rank(2)
    rank_two_second = source.fork_for_rank(2)
    rank_three = source.fork_for_rank(3)
    assert rank_two_first.digest() == rank_two_second.digest()
    assert rank_two_first.digest() != source.digest()
    assert rank_two_first.digest() != rank_three.digest()


def test_committed_boundary_is_a_deep_snapshot() -> None:
    state = _state(3)
    cursor = _cursor(3)
    boundary = capture_committed_boundary(state, cursor, RNGState.capture())
    state.loss_weights["ntp"] = 9.0
    cursor.extra["changed"] = True
    assert boundary.trainer_state.loss_weights == {"ntp": 1.0}
    assert boundary.data_cursor.extra == {}


def test_atomic_checkpoint_round_trip_and_manifest(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    assert manager.find_latest_valid() is None
    assert manager.find_latest_valid_with_metadata() is None
    path = _save(manager, 12)

    assert (path / "COMPLETE").is_file()
    assert (path / "manifest.json").is_file()
    assert (tmp_path / "latest").read_text().strip() == path.name
    assert manager.inspect(path)["global_step"] == 12
    loaded = manager.load(
        {},
        expected_critical_fingerprint={"experts": 8, "lr": 1e-4},
        expected_data_fingerprint={"manifest": "abc"},
        expected_run_id="unit-test",
        expected_global_batch_tokens=4096,
    )
    assert loaded.path == path
    assert loaded.stateful["model"]["weight"] == 12
    assert loaded.trainer_state.global_step == 12
    assert loaded.data_cursor.global_token_index == 12 * 4096
    # v1-v3 checkpoints predate data-mode and optimizer audit metadata. Their
    # authenticated empty ``extra`` mapping must remain fully loadable.
    assert loaded.metadata["extra"] == {}


def test_latest_valid_path_and_metadata_are_resolved_together(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    path = _save(manager, 7)

    resolved = manager.find_latest_valid_with_metadata()

    assert resolved is not None
    resolved_path, metadata = resolved
    assert resolved_path == path
    assert metadata["global_step"] == 7


def test_save_and_prune_hash_each_unchanged_state_payload_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashed_state_files: list[Path] = []
    original_sha256_file = checkpoint_module._sha256_file

    def tracking_sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        if path.name == "state.pkl":
            hashed_state_files.append(path)
        return original_sha256_file(path, chunk_size)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", tracking_sha256_file)
    manager = CheckpointManager(
        tmp_path,
        backend=PickleBackend(),
        keep_periodic=3,
    )

    for step in range(1, 5):
        _save(manager, step)

    # Each new payload is hashed while its manifest is built. The immediate
    # verify and every retention scan reuse that same manager's trusted proof.
    assert len(hashed_state_files) == 4
    assert all(path.parent.parent.name.endswith(".incomplete") for path in hashed_state_files)


def test_new_manager_must_hash_existing_checkpoint_before_using_its_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = CheckpointManager(tmp_path, backend=PickleBackend())
    path = _save(writer, 1)
    state_hashes = 0
    original_sha256_file = checkpoint_module._sha256_file

    def tracking_sha256_file(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        nonlocal state_hashes
        if file_path.name == "state.pkl":
            state_hashes += 1
        return original_sha256_file(file_path, chunk_size)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", tracking_sha256_file)
    reader = CheckpointManager(tmp_path, backend=PickleBackend())

    assert reader.verify(path)["global_step"] == 1
    assert state_hashes == 1
    assert reader.verify(path)["global_step"] == 1
    assert state_hashes == 1


def test_changed_authenticated_identity_cannot_reuse_prior_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    path = _save(manager, 1)

    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["extra"]["rewritten"] = True
    metadata_bytes = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    metadata_path.write_bytes(metadata_bytes)

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metadata.json"] = hashlib.sha256(metadata_bytes).hexdigest()
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    (path / "COMPLETE").write_text(f"{hashlib.sha256(manifest_bytes).hexdigest()}\n")

    state_hashes = 0
    original_sha256_file = checkpoint_module._sha256_file

    def tracking_sha256_file(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        nonlocal state_hashes
        if file_path.name == "state.pkl":
            state_hashes += 1
        return original_sha256_file(file_path, chunk_size)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", tracking_sha256_file)

    assert manager.verify(path)["extra"]["rewritten"] is True
    assert state_hashes == 1


def test_partial_accumulation_rolls_back_cursor_rng_and_metadata(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    committed_state = _state(4)
    committed_cursor = _cursor(4)
    committed_rng = RNGState.capture()
    boundary = capture_committed_boundary(committed_state, committed_cursor, committed_rng)
    partial_state = _state(4, micro_step=2)
    advanced_cursor = _cursor(4)
    advanced_cursor.global_token_index += 2048

    manager.save(
        {"model": {"weight": 4}},
        trainer_state=partial_state,
        data_cursor=advanced_cursor,
        rng_state=RNGState.capture(),
        committed_boundary=boundary,
        critical_fingerprint="critical",
        data_fingerprint="data",
        kind="interrupt",
    )
    loaded = manager.load(
        {},
        expected_critical_fingerprint="critical",
        expected_data_fingerprint="data",
    )
    assert loaded.metadata["rollback_applied"] is True
    assert loaded.trainer_state.micro_step_in_accumulation == 0
    assert loaded.data_cursor.global_token_index == committed_cursor.global_token_index
    assert loaded.rng_state.digest() == committed_rng.digest()


def test_added_rank_forks_saved_rng_instead_of_duplicating_it(tmp_path: Path) -> None:
    source_manager = CheckpointManager(tmp_path, backend=PickleBackend())
    _save(source_manager, 1)
    saved = source_manager.load({}).rng_state

    class LocalOnlyManager(CheckpointManager):
        def _resolve_for_load(
            self, checkpoint: str | Path
        ) -> tuple[Path, dict[str, Any]]:
            return self._resolve_verified(checkpoint)

        def _synchronize_error(self, local_error: Exception | None, context: str) -> None:
            del context
            if local_error is not None:
                raise local_error

    added_rank = LocalOnlyManager(
        tmp_path,
        backend=PickleBackend(),
        rank=1,
        world_size=2,
    ).load({})
    assert added_rank.source_rank == 0
    assert added_rank.rng_forked
    assert added_rank.rng_state.digest() != saved.digest()


def test_partial_accumulation_without_boundary_is_refused(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    with pytest.raises(ValueError, match="committed boundary"):
        manager.save(
            {"model": {}},
            trainer_state=_state(1, micro_step=1),
            data_cursor=_cursor(1),
            critical_fingerprint="critical",
            data_fingerprint="data",
        )


def test_fingerprint_mismatch_is_detected_before_mutating_targets(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    _save(manager, 1)
    target = {"sentinel": 1}
    with pytest.raises(IncompatibleCheckpointError, match="configuration"):
        manager.load(target, expected_critical_fingerprint={"lr": 2e-4, "experts": 8})
    assert target == {"sentinel": 1}


def test_explicit_relative_checkpoint_path_is_relative_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / "base-v1"
    manager = CheckpointManager(run_dir, backend=PickleBackend())
    checkpoint = _save(manager, 1)
    monkeypatch.chdir(workspace)

    assert manager.resolve(Path("runs/base-v1") / checkpoint.name) == checkpoint
    assert manager.resolve(checkpoint.name) == checkpoint


def test_corrupt_latest_falls_back_to_last_complete_checkpoint(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    first = _save(manager, 1)
    second = _save(manager, 2)
    with (second / "state" / "state.pkl").open("ab") as handle:
        handle.write(b"corruption")

    assert manager.resolve("auto") == first
    with pytest.raises(CheckpointCorruptError):
        manager.verify(second)


def test_fallback_can_replay_and_replace_a_quarantined_corrupt_step(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    first = _save(manager, 1)
    corrupt = _save(manager, 2)
    with (corrupt / "state" / "state.pkl").open("ab") as handle:
        handle.write(b"corruption")

    assert manager.resolve("auto") == first
    replacement = _save(manager, 2)
    assert replacement == corrupt
    assert manager.verify(replacement)["global_step"] == 2
    assert any(path.name.startswith(".corrupt-step-000000000002-periodic-") for path in tmp_path.iterdir())


def test_garbage_latest_pointer_falls_back_to_newest_valid_directory(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    _save(manager, 1)
    newest = _save(manager, 2)

    # ``latest`` is only a convenience pointer. A torn/non-UTF8 pointer must
    # not hide a fully committed checkpoint directory.
    (tmp_path / "latest").write_bytes(b"\xfftorn-pointer\x00\n")
    assert manager.resolve("auto") == newest


def test_same_step_interrupt_sequence_survives_resume(tmp_path: Path) -> None:
    manager = CheckpointManager(
        tmp_path,
        backend=PickleBackend(),
        keep_interrupt=2,
    )
    critical = {"lr": 1e-4, "experts": 8}
    data = {"manifest": "abc"}

    committed_state = _state(4)
    committed_cursor = _cursor(4)
    first_boundary = capture_committed_boundary(committed_state, committed_cursor)
    first_partial = _state(4, micro_step=2)
    first_partial.extra["checkpoint_request_sequence"] = 1
    first_boundary.trainer_state.extra["checkpoint_request_sequence"] = 1
    first = manager.save(
        {"model": {"weight": 4}},
        trainer_state=first_partial,
        data_cursor=committed_cursor,
        committed_boundary=first_boundary,
        critical_fingerprint=critical,
        data_fingerprint=data,
        kind="interrupt",
        tag="request-000001",
    )

    resumed = manager.load(
        {},
        expected_critical_fingerprint=critical,
        expected_data_fingerprint=data,
    )
    assert resumed.trainer_state.global_step == 4
    assert resumed.trainer_state.extra["checkpoint_request_sequence"] == 1

    # A second USR1/interrupt may arrive after resume but before step 4 commits.
    # Persisting the request sequence must give it a distinct immutable name.
    second_boundary = capture_committed_boundary(
        resumed.trainer_state,
        resumed.data_cursor,
        resumed.rng_state,
    )
    second_partial = resumed.trainer_state.clone()
    second_partial.micro_step_in_accumulation = 1
    second_partial.extra["checkpoint_request_sequence"] = 2
    second_boundary.trainer_state.extra["checkpoint_request_sequence"] = 2
    second = manager.save(
        {"model": {"weight": 4}},
        trainer_state=second_partial,
        data_cursor=resumed.data_cursor,
        committed_boundary=second_boundary,
        critical_fingerprint=critical,
        data_fingerprint=data,
        kind="interrupt",
        tag="request-000002",
    )

    assert first != second
    assert first.is_dir() and second.is_dir()
    latest = manager.load(
        {},
        expected_critical_fingerprint=critical,
        expected_data_fingerprint=data,
    )
    assert latest.path == second
    assert latest.trainer_state.extra["checkpoint_request_sequence"] == 2


def test_stale_but_valid_latest_pointer_does_not_hide_newer_commit(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    first = _save(manager, 1)
    second = _save(manager, 2)
    (tmp_path / "latest").write_text(first.name)
    assert manager.find_latest() == second


def test_incomplete_directories_are_never_resume_candidates(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    complete = _save(manager, 1)
    incomplete = tmp_path / ".step-999999999999-periodic.incomplete"
    incomplete.mkdir()
    (tmp_path / "latest").write_text(incomplete.name)
    assert manager.resolve() == complete


def test_retry_replaces_only_the_same_incomplete_staging_directory(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    stale = tmp_path / ".step-000000000003-periodic.incomplete"
    stale.mkdir(parents=True)
    (stale / "torn-shard").write_bytes(b"partial")
    complete = _save(manager, 3)
    assert complete.is_dir()
    assert not stale.exists()


def test_complete_checkpoint_is_never_silently_overwritten(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, backend=PickleBackend())
    complete = _save(manager, 3)
    complete_marker = (complete / "COMPLETE").read_text()
    with pytest.raises(RuntimeError, match="overwrite"):
        _save(manager, 3)
    assert (complete / "COMPLETE").read_text() == complete_marker


def test_retention_keeps_last_three_periodic_one_interrupt_and_milestones(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path,
        backend=PickleBackend(),
        keep_periodic=3,
        keep_interrupt=1,
    )
    periodic = [_save(manager, step) for step in range(1, 5)]
    interrupts = [_save(manager, step, "interrupt") for step in range(5, 7)]
    milestones = [_save(manager, step, "milestone") for step in range(7, 9)]

    assert not periodic[0].exists()
    assert all(path.exists() for path in periodic[1:])
    assert not interrupts[0].exists()
    assert interrupts[1].exists()
    assert all(path.exists() for path in milestones)


def test_corrupt_checkpoint_does_not_consume_a_retention_slot(tmp_path: Path) -> None:
    manager = CheckpointManager(
        tmp_path,
        backend=PickleBackend(),
        keep_periodic=3,
    )
    second = _save(manager, 2)
    third = _save(manager, 3)
    fourth = _save(manager, 4)
    with (fourth / "state" / "state.pkl").open("ab") as handle:
        handle.write(b"corruption")

    fifth = _save(manager, 5)
    assert second.exists() and third.exists() and fourth.exists() and fifth.exists()
    assert [metadata["global_step"] for _, metadata in manager.valid_checkpoints()] == [2, 3, 5]


def test_run_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    first = RunLock(tmp_path / ".run.lock").acquire()
    try:
        with pytest.raises(RunLockedError):
            RunLock(tmp_path / ".run.lock").acquire()
    finally:
        first.release()


def test_manager_context_holds_lock_between_checkpoints(tmp_path: Path) -> None:
    first = CheckpointManager(tmp_path, backend=PickleBackend())
    second = CheckpointManager(tmp_path, backend=PickleBackend())
    with first:
        assert first.run_lock_held
        with pytest.raises(RunLockedError):
            second.acquire_run_lock()
    assert not first.run_lock_held


def test_stable_fingerprint_is_order_independent() -> None:
    assert stable_fingerprint({"b": 2, "a": [1]}) == stable_fingerprint(
        {"a": [1], "b": 2}
    )


def test_stable_fingerprint_preserves_a_precomputed_sha256() -> None:
    digest = "a" * 64
    assert stable_fingerprint(digest) == digest
