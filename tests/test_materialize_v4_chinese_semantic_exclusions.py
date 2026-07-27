from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from twen.data.audits import DataAuditError
from twen.data.sources import DataSourceError
from twen.utils import sha256_file


def _load_file_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).parents[1]
materializer = _load_file_module(
    "materialize_v4_chinese_semantic_exclusions",
    _ROOT / "scripts" / "materialize_v4_chinese_semantic_exclusions.py",
)
_phase_test_helpers = _load_file_module(
    "_materializer_phase_exclusion_test_helpers",
    _ROOT / "tests" / "test_phase_exclusion.py",
)
_write_audit = _phase_test_helpers._write_audit
_write_extracted = _phase_test_helpers._write_extracted


@dataclass
class _Fixture:
    root: Path
    manifest: Path
    audit: Path
    ledger: Path
    scanner: Path
    validation_bytes: bytes
    rows: list[dict[str, Any]]
    arguments: dict[str, Any]


def _write_materializer_audit(root: Path, manifest: Path) -> Path:
    attestation = _write_audit(root, manifest)
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["gates"]["deterministic_content_quality_scan"] = {
        "passed": True,
        "status": "complete_policy_v1_no_findings",
    }
    value.pop("attestation_fingerprint")
    value["attestation_fingerprint"] = materializer._canonical_sha256(value)
    materializer.atomic_write_json(attestation, value)
    complete_path = attestation.parent / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.update(
        {
            "attestation_sha256": sha256_file(attestation),
            "attestation_fingerprint": value["attestation_fingerprint"],
        }
    )
    materializer.atomic_write_json(complete_path, complete)
    return attestation


def _scanner(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "import re",
                "_CONVERSION_MARKERS = (",
                "    ('appeared_substitution', re.compile(r'BADMARK')),",
                ")",
                "_MALFORMED_PUNCTUATION = re.compile(r'!。')",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ordered = sorted(rows, key=materializer._ledger_sort_key)
    with path.open("wb") as handle:
        for row in ordered:
            handle.write(materializer._canonical_bytes(row) + b"\n")
    return path


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train: list[tuple[str, str, int]] | None = None,
) -> _Fixture:
    scanner = _scanner(tmp_path / "scanner.py")
    monkeypatch.setattr(materializer, "SEMANTIC_SCANNER_PATH", scanner)
    source_id = "shared_source"
    validation = [("validation", "byte-preserved validation 文本", 7)]
    train_rows = train or [
        ("conversion", "ordinary BADMARK semantic artifact", 10),
        ("punctuation", "broken!。 punctuation", 11),
        ("clean", "clean retained document", 20),
    ]
    manifest = _write_extracted(
        tmp_path / "parent",
        source_id=source_id,
        train=train_rows,
        validation=validation,
    )
    parent_value = json.loads(manifest.read_text(encoding="utf-8"))
    parent_value["audits"] = {
        "accepted_row_provenance_ledger": "complete",
        "code_license_allowlist": "complete",
        "cross_source_exact_dedup": "pending_parent_reaudit",
        "stable_train_validation_split": "complete",
    }
    materializer.atomic_write_json(manifest, parent_value)
    complete_path = manifest.parent / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["manifest_sha256"] = sha256_file(manifest)
    materializer.atomic_write_json(complete_path, complete)
    audit = _write_materializer_audit(tmp_path / "audit", manifest)
    phase = materializer._authenticate_input(
        parent_manifest=manifest,
        base_audit=audit,
        expected_parent_manifest_sha256=sha256_file(manifest),
        expected_base_audit_sha256=sha256_file(audit),
    )
    scan = materializer._load_scanner(expected_sha256=sha256_file(scanner))
    rebuild = tmp_path / "rebuild"
    rebuild.mkdir()
    projection = materializer._project_phase(
        phase=phase,
        phase_name="primary",
        source_id=source_id,
        scanner=scan,
        work=rebuild,
    )
    rows = projection.rebuilt_ledger_rows
    excluded_tokens = sum(row["token_count"] for row in rows)
    retained_tokens = sum(projection.retained_tokens["train"].values())
    ledger = _write_rows(tmp_path / "ledger.jsonl", rows)
    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    validation_file = manifest.parent / raw_manifest["validation_files"][0]["path"]
    arguments: dict[str, Any] = {
        "parent_manifest": manifest,
        "base_audit": audit,
        "canonical_ledger": ledger,
        "phase": "primary",
        "source_id": source_id,
        "expected_parent_manifest_sha256": sha256_file(manifest),
        "expected_base_audit_sha256": sha256_file(audit),
        "expected_ledger_sha256": sha256_file(ledger),
        "expected_exclusion_count": len(rows),
        "expected_excluded_tokens": excluded_tokens,
        "expected_scanner_sha256": sha256_file(scanner),
        "expected_normalizer_sha256": sha256_file(materializer.NORMALIZER_SOURCE_PATH),
        "required_source_tokens": {source_id: retained_tokens},
        "required_aggregate_tokens": retained_tokens,
        "formal_training_tokens": retained_tokens - 1,
        "global_batch_tokens": 1,
        "output_root": tmp_path / "output",
    }
    return _Fixture(
        root=tmp_path,
        manifest=manifest,
        audit=audit,
        ledger=ledger,
        scanner=scanner,
        validation_bytes=validation_file.read_bytes(),
        rows=rows,
        arguments=arguments,
    )


def _mutated_ledger(
    fixture: _Fixture,
    rows: list[dict[str, Any]],
    *,
    name: str,
) -> tuple[Path, str]:
    path = _write_rows(fixture.root / name, rows)
    return path, sha256_file(path)


def _refingerprint(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("ledger_entry_fingerprint", None)
    value["ledger_entry_fingerprint"] = materializer._canonical_sha256(value)
    return value


def _run(
    fixture: _Fixture,
    **overrides: Any,
) -> Path:
    arguments = {**fixture.arguments, **overrides}
    return materializer.materialize_semantic_exclusions(**arguments)


def _inventory_path(fixture: _Fixture, *, role: str) -> Path:
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    return (fixture.manifest.parent / manifest[f"{role}_files"][0]["path"]).resolve()


def _rewrite_parent_attribution(
    fixture: _Fixture,
    rows: list[dict[str, Any]],
) -> Path:
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    relative = manifest["attribution_files"][0]["path"]
    attribution = fixture.manifest.parent / relative
    attribution.write_bytes(
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    )
    identity = {
        "path": relative,
        "size": attribution.stat().st_size,
        "sha256": sha256_file(attribution),
    }
    manifest["attribution_files"] = [identity]
    for source in manifest["sources"]:
        for chunk in source["chunks"]:
            chunk["outputs"] = [
                identity if output["path"] == relative else output for output in chunk["outputs"]
            ]
    manifest_identity = {field: manifest[field] for field in materializer._MANIFEST_IDENTITY_FIELDS}
    manifest["corpus_fingerprint"] = materializer._manifest_fingerprint(manifest_identity)
    materializer.atomic_write_json(fixture.manifest, manifest)
    complete_path = fixture.manifest.parent / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["corpus_fingerprint"] = manifest["corpus_fingerprint"]
    complete["manifest_sha256"] = sha256_file(fixture.manifest)
    materializer.atomic_write_json(complete_path, complete)
    return attribution


def _reaudit_parent(fixture: _Fixture, root: Path) -> None:
    audit = _write_materializer_audit(root, fixture.manifest)
    fixture.audit = audit
    fixture.arguments.update(
        {
            "base_audit": audit,
            "expected_parent_manifest_sha256": sha256_file(fixture.manifest),
            "expected_base_audit_sha256": sha256_file(audit),
        }
    )


def _install_transient_binary_stream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    active: dict[str, bool],
) -> None:
    original_open = Path.open
    transient = target.read_bytes().replace(b"\n", b" \n")
    assert transient != target.read_bytes()

    def open_with_transient_bytes(
        self: Path,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if active["enabled"] and mode == "rb" and self.resolve() == target:
            return io.BytesIO(transient)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_transient_bytes)


def test_materializes_exact_projection_and_blocks_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    manifest_path = _run(fixture)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["actual_train_documents"] == 1
    assert manifest["actual_train_tokens"] == 20
    assert manifest["semantic_excluded_train_documents"] == 2
    assert manifest["semantic_excluded_train_tokens"] == 21
    assert manifest["ready_for_training"] is False
    assert manifest["requires_independent_reaudit"] is True
    assert manifest["authorizes_training"] is False
    parent = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    for name, status in parent["audits"].items():
        expected = (
            materializer._PENDING_BASE_REAUDIT_STATUS
            if name in materializer._BASE_REAUDIT_GATE_NAMES
            else status
        )
        assert manifest["audits"][name] == expected
    assert all(
        manifest["audits"][name] == materializer._PENDING_BASE_REAUDIT_STATUS
        for name in materializer._BASE_REAUDIT_GATE_NAMES
    )
    assert manifest["audits"]["chinese_semantic_exclusion"] == (
        "complete_authenticated_canonical_projection"
    )
    assert manifest["audits"]["validation_byte_preservation"] == ("complete_sha256_identical")
    audit_projection = manifest["materialization_audit"]["chinese_semantic_exclusion"][
        "audit_projection"
    ]
    assert audit_projection["algorithm"] == (materializer._AUDIT_PROJECTION_ALGORITHM)
    assert audit_projection["preserved_parent_statuses"] == {
        name: status
        for name, status in sorted(parent["audits"].items())
        if name not in materializer._BASE_REAUDIT_GATE_NAMES
    }
    train = manifest_path.parent / manifest["train_files"][0]["path"]
    assert [json.loads(line)["text"] for line in train.read_text().splitlines()] == [
        "clean retained document"
    ]
    validation = manifest_path.parent / manifest["validation_files"][0]["path"]
    assert validation.read_bytes() == fixture.validation_bytes
    attribution = manifest_path.parent / manifest["attribution_files"][0]["path"]
    attribution_rows = [json.loads(line) for line in attribution.read_text().splitlines()]
    assert [row["split"] for row in attribution_rows] == ["train", "validation"]
    output_ledger = manifest_path.parent / materializer.OUTPUT_LEDGER_NAME
    assert output_ledger.read_bytes() == fixture.ledger.read_bytes()
    attestation = materializer.validate_semantic_exclusion_output(manifest_path)
    assert attestation["metrics"]["excluded_train_documents"] == 2
    assert attestation["capacity"]["aggregate"]["passed"] is True
    assert not (manifest_path.parent / "rejections.jsonl").exists()


def test_audit_projection_rejects_unresolved_static_parent_status() -> None:
    with pytest.raises(
        DataAuditError,
        match="unresolved non-Base audit statuses",
    ):
        materializer._project_parent_audits(
            {
                "code_license_allowlist": "pending_manual_review",
                "cross_source_exact_dedup": "complete_no_matches",
            },
            {name: {"status": "complete"} for name in materializer._BASE_REAUDIT_GATE_NAMES},
        )


def test_audit_projection_uses_authenticated_gate_inventory_and_preserves_cooldown() -> None:
    parent = {
        "accepted_row_provenance_ledger": "complete",
        "cross_phase_exact_duplicate_exclusion": "complete_authenticated_projection",
        "cross_phase_near_duplicate_exclusion": ("complete_exhaustive_attested_locator_projection"),
        "cross_source_exact_dedup": "complete_no_matches",
    }
    gates = {name: {"status": f"complete_{name}"} for name in materializer._BASE_REAUDIT_GATE_NAMES}
    projected, evidence = materializer._project_parent_audits(parent, gates)
    assert projected["cross_phase_exact_duplicate_exclusion"] == (
        "complete_authenticated_projection"
    )
    assert projected["cross_phase_near_duplicate_exclusion"] == (
        "complete_exhaustive_attested_locator_projection"
    )
    assert evidence["authenticated_base_audit_gate_statuses"] == {
        name: gates[name]["status"] for name in sorted(gates)
    }
    assert all(projected[name] == materializer._PENDING_BASE_REAUDIT_STATUS for name in gates)


def test_audit_projection_rejects_unexpected_authenticated_gate_inventory() -> None:
    with pytest.raises(
        DataAuditError,
        match="gate inventory differs",
    ):
        materializer._project_parent_audits(
            {"code_license_allowlist": "complete"},
            {"future_dynamic_gate": {"status": "complete"}},
        )


def test_validator_rejects_refingerprinted_audit_projection_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    manifest_path = _run(fixture)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audits"]["code_license_allowlist"] = "complete_tampered"
    materializer.atomic_write_json(manifest_path, manifest)

    attestation_path = root / materializer.ATTESTATION_NAME
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["output"]["manifest_sha256"] = sha256_file(manifest_path)
    attestation.pop("attestation_fingerprint")
    attestation["attestation_fingerprint"] = materializer._canonical_sha256(attestation)
    materializer.atomic_write_json(attestation_path, attestation)

    complete_path = root / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["manifest_sha256"] = sha256_file(manifest_path)
    complete["semantic_exclusion_attestation"].update(
        {
            "size": attestation_path.stat().st_size,
            "sha256": sha256_file(attestation_path),
            "attestation_fingerprint": attestation["attestation_fingerprint"],
        }
    )
    materializer.atomic_write_json(complete_path, complete)
    with pytest.raises(
        DataAuditError,
        match="parent audit projection differs",
    ):
        materializer.validate_semantic_exclusion_output(manifest_path)


def test_attribution_binding_does_not_retain_raw_payload() -> None:
    assert "raw" not in materializer._AttributionBinding.__slots__


def test_attribution_duplicate_text_bucket_preserves_fifo_one_to_one() -> None:
    first = materializer._AttributionBinding(
        record="first",
        inventory_sha256="1" * 64,
    )
    second = materializer._AttributionBinding(
        record="second",
        inventory_sha256="1" * 64,
    )
    queues = {
        ("train", "source"): {
            "2" * 64: materializer.deque((first, second)),
        }
    }
    assert materializer._remaining_attribution(queues) == 2
    assert (
        materializer._pop_attribution(
            queues,
            role="train",
            source_id="source",
            text_sha256="2" * 64,
        )
        is first
    )
    assert materializer._remaining_attribution(queues) == 1
    assert (
        materializer._pop_attribution(
            queues,
            role="train",
            source_id="source",
            text_sha256="2" * 64,
        )
        is second
    )
    assert queues == {}


def test_distinct_train_texts_may_share_parent_stable_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        train=[
            ("shared-stable", "ordinary BADMARK semantic artifact", 10),
            ("shared-stable", "clean retained document", 20),
        ],
    )
    attribution = _inventory_path(fixture, role="attribution")
    rows = [json.loads(line) for line in attribution.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["stable_id"] == rows[1]["stable_id"]
    assert rows[0]["text_sha256"] != rows[1]["text_sha256"]
    manifest_path = _run(fixture)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["actual_train_documents"] == 1
    assert manifest["semantic_excluded_train_documents"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "does not cover corpus document exactly"),
        ("extra", "no exact corpus document for 1 rows"),
    ],
)
def test_parent_attribution_cardinality_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    attribution = _inventory_path(fixture, role="attribution")
    rows = [json.loads(line) for line in attribution.read_text(encoding="utf-8").splitlines()]
    if mutation == "missing":
        del rows[2]
    else:
        rows.append(dict(rows[2]))
    _rewrite_parent_attribution(fixture, rows)
    _reaudit_parent(fixture, tmp_path / "reaudit")
    with pytest.raises(DataAuditError, match=message):
        _run(fixture)
    assert not Path(fixture.arguments["output_root"]).exists()


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("expected_parent_manifest_sha256", "0" * 64, "parent manifest external pin"),
        ("expected_base_audit_sha256", "0" * 64, "base audit external pin"),
        ("expected_ledger_sha256", "0" * 64, "ledger external pin"),
        ("expected_scanner_sha256", "0" * 64, "scanner external pin"),
        ("expected_normalizer_sha256", "0" * 64, "normalizer external pin"),
    ],
)
def test_external_identity_pins_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(DataAuditError, match=message):
        _run(fixture, **{argument: value})


def test_omitted_scanner_hit_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    rows = fixture.rows[:1]
    ledger, digest = _mutated_ledger(fixture, rows, name="omitted.jsonl")
    with pytest.raises(DataAuditError, match="omitted=1"):
        _run(
            fixture,
            canonical_ledger=ledger,
            expected_ledger_sha256=digest,
            expected_exclusion_count=1,
            expected_excluded_tokens=rows[0]["token_count"],
        )


def test_extra_ledger_hit_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    extra = dict(fixture.rows[-1])
    extra.update(
        {
            "path": "extracted/shared_source/chunk-999999/train.jsonl",
            "line_number": 999,
            "stable_id": hashlib.sha256(b"extra-stable").hexdigest(),
            "attribution_line_number": 999,
        }
    )
    extra = _refingerprint(extra)
    rows = [*fixture.rows, extra]
    ledger, digest = _mutated_ledger(fixture, rows, name="extra.jsonl")
    with pytest.raises(DataAuditError, match="extra=1"):
        _run(
            fixture,
            canonical_ledger=ledger,
            expected_ledger_sha256=digest,
            expected_exclusion_count=3,
            expected_excluded_tokens=sum(row["token_count"] for row in rows),
        )


def test_duplicate_ledger_location_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    rows = [fixture.rows[0], fixture.rows[0], fixture.rows[1]]
    ledger, digest = _mutated_ledger(fixture, rows, name="duplicate.jsonl")
    with pytest.raises(DataAuditError, match="unsorted or duplicated"):
        _run(
            fixture,
            canonical_ledger=ledger,
            expected_ledger_sha256=digest,
            expected_exclusion_count=3,
            expected_excluded_tokens=31,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_sha",
        "normalized_sha",
        "stable_id",
        "token_count",
        "reason",
        "shard_sha",
        "location",
        "attribution_location",
    ],
)
def test_stale_or_forged_ledger_fields_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    rows = [dict(row) for row in fixture.rows]
    row = rows[0]
    if mutation == "raw_sha":
        row["text_sha256"] = "1" * 64
    elif mutation == "normalized_sha":
        row["normalized_text_sha256"] = "2" * 64
    elif mutation == "stable_id":
        row["stable_id"] = "3" * 64
    elif mutation == "token_count":
        row["token_count"] += 1
    elif mutation == "reason":
        row["reasons"] = ["fabricated_reason"]
        row["reason_occurrences"] = {"fabricated_reason": 1}
    elif mutation == "shard_sha":
        row["shard_sha256"] = "4" * 64
    elif mutation == "location":
        row["line_number"] += 100
    elif mutation == "attribution_location":
        row["attribution_line_number"] += 100
    rows[0] = _refingerprint(row)
    ledger, digest = _mutated_ledger(
        fixture,
        rows,
        name=f"stale-{mutation}.jsonl",
    )
    expected_tokens = sum(item["token_count"] for item in rows)
    expected_error = "omitted=1, extra=1" if mutation == "location" else "stale_or_field_mismatch=1"
    with pytest.raises(DataAuditError, match=expected_error):
        _run(
            fixture,
            canonical_ledger=ledger,
            expected_ledger_sha256=digest,
            expected_excluded_tokens=expected_tokens,
        )


def test_parent_content_change_after_audit_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    train = fixture.manifest.parent / manifest["train_files"][0]["path"]
    train.write_bytes(train.read_bytes() + b"\n")
    with pytest.raises(DataAuditError, match="invalid extracted phase input"):
        _run(fixture)


def test_transient_manifest_replacement_between_authenticated_reads_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_payload = fixture.manifest.read_bytes()
    transient_value = json.loads(original_payload)
    transient_value["profile"] = "transient-unpinned-profile"
    transient_payload = json.dumps(
        transient_value,
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    original_validate = materializer.phase_helpers.validate_extracted_base_corpus
    original_verify = materializer.phase_helpers._verify_extracted_owned_files

    def validate_then_replace(*args: Any, **kwargs: Any) -> Any:
        result = original_validate(*args, **kwargs)
        fixture.manifest.write_bytes(transient_payload)
        return result

    def verify_then_restore(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_verify(*args, **kwargs)
        finally:
            fixture.manifest.write_bytes(original_payload)

    monkeypatch.setattr(
        materializer.phase_helpers,
        "validate_extracted_base_corpus",
        validate_then_replace,
    )
    monkeypatch.setattr(
        materializer.phase_helpers,
        "_verify_extracted_owned_files",
        verify_then_restore,
    )
    with pytest.raises(
        DataAuditError,
        match="changed between authentication and consumption",
    ):
        _run(fixture)
    assert fixture.manifest.read_bytes() == original_payload
    assert not Path(fixture.arguments["output_root"]).exists()


def test_transient_train_consumption_stream_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    active = {"enabled": False}
    _install_transient_binary_stream(
        monkeypatch,
        target=_inventory_path(fixture, role="train"),
        active=active,
    )
    original = materializer._copy_documents

    def copy_with_transient_stream(**kwargs: Any) -> Any:
        if kwargs["role"] != "train":
            return original(**kwargs)
        active["enabled"] = True
        try:
            return original(**kwargs)
        finally:
            active["enabled"] = False

    monkeypatch.setattr(
        materializer,
        "_copy_documents",
        copy_with_transient_stream,
    )
    with pytest.raises(
        DataAuditError,
        match="parent train consumed stream identity differs",
    ):
        _run(fixture)
    assert not Path(fixture.arguments["output_root"]).exists()


def test_transient_attribution_queue_stream_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    active = {"enabled": False}
    _install_transient_binary_stream(
        monkeypatch,
        target=_inventory_path(fixture, role="attribution"),
        active=active,
    )
    original = materializer._attribution_queues

    def queue_with_transient_stream(phase: Any) -> Any:
        active["enabled"] = True
        try:
            return original(phase)
        finally:
            active["enabled"] = False

    monkeypatch.setattr(
        materializer,
        "_attribution_queues",
        queue_with_transient_stream,
    )
    with pytest.raises(
        DataAuditError,
        match="parent attribution consumed stream identity differs",
    ):
        _run(fixture)
    assert not Path(fixture.arguments["output_root"]).exists()


def test_transient_attribution_filter_stream_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    active = {"enabled": False}
    _install_transient_binary_stream(
        monkeypatch,
        target=_inventory_path(fixture, role="attribution"),
        active=active,
    )
    original = materializer._write_filtered_attribution

    def filter_with_transient_stream(**kwargs: Any) -> Any:
        active["enabled"] = True
        try:
            return original(**kwargs)
        finally:
            active["enabled"] = False

    monkeypatch.setattr(
        materializer,
        "_write_filtered_attribution",
        filter_with_transient_stream,
    )
    with pytest.raises(
        DataAuditError,
        match="parent attribution consumed stream identity differs",
    ):
        _run(fixture)
    assert not Path(fixture.arguments["output_root"]).exists()


def test_scanner_change_during_materialization_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original = materializer._project_phase

    def mutate_scanner(**kwargs: Any) -> Any:
        result = original(**kwargs)
        fixture.scanner.write_text(
            fixture.scanner.read_text(encoding="utf-8") + "# drift\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(materializer, "_project_phase", mutate_scanner)
    with pytest.raises(
        DataAuditError,
        match=r"source or input identity changed|external pin mismatch",
    ):
        _run(fixture)
    assert not Path(fixture.arguments["output_root"]).exists()


def test_current_real_scanner_exact_payload_is_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner_path = (_ROOT / "scripts" / "audit_v4_chinese_semantic_noise.py").resolve()
    scanner_sha256 = sha256_file(scanner_path)
    assert scanner_sha256 == ("f739f359cf44925cddb3a59bfbb6890b84997f73500db1b97efe308b93122d32")
    module_name = f"_twen_semantic_scanner_{scanner_sha256}"
    sentinel = materializer.ModuleType("_scanner_test_sentinel")
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    sys.modules[module_name] = sentinel
    monkeypatch.setattr(
        materializer,
        "SEMANTIC_SCANNER_PATH",
        scanner_path,
    )
    try:
        scanner = materializer._load_scanner(
            expected_sha256=scanner_sha256,
        )
        assert sys.modules[module_name] is sentinel
    finally:
        if previous_module is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    assert len(scanner.conversion_markers) == 6
    markers = dict(scanner.conversion_markers)
    assert markers["engine_substitution"].findall("发念头") == ["发念头"]
    assert scanner.malformed_punctuation.findall("测试\uff01。") == ["\uff01。"]


def test_scanner_path_swap_cannot_execute_then_restore_unpinned_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = _scanner(tmp_path / "scanner.py").resolve()
    pinned_payload = scanner.read_bytes()
    encoded_pinned = base64.b64encode(pinned_payload).decode()
    transient_payload = (
        "import base64\n"
        "import re\n"
        "from pathlib import Path\n"
        "_CONVERSION_MARKERS = (\n"
        "    ('transient_poison', re.compile(r'.+')),\n"
        ")\n"
        "_MALFORMED_PUNCTUATION = re.compile(r'(?!)')\n"
        f"Path(__file__).write_bytes(base64.b64decode({encoded_pinned!r}))\n"
    ).encode()
    original_read_bytes = Path.read_bytes
    armed = True

    def read_then_swap(self: Path) -> bytes:
        nonlocal armed
        payload = original_read_bytes(self)
        if armed and self.resolve() == scanner:
            armed = False
            scanner.write_bytes(transient_payload)
        return payload

    monkeypatch.setattr(materializer, "SEMANTIC_SCANNER_PATH", scanner)
    monkeypatch.setattr(Path, "read_bytes", read_then_swap)
    try:
        with pytest.raises(
            DataAuditError,
            match="changed before exact-payload execution",
        ):
            materializer._load_scanner(expected_sha256=hashlib.sha256(pinned_payload).hexdigest())
        assert scanner.read_bytes() == transient_payload
    finally:
        scanner.write_bytes(pinned_payload)


def test_validation_change_during_projection_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original = materializer._copy_documents

    def mutate_validation(**kwargs: Any) -> Any:
        result = original(**kwargs)
        if kwargs["role"] == "validation":
            output = kwargs["work"] / result[0]["path"]
            output.write_bytes(output.read_bytes() + b" ")
        return result

    monkeypatch.setattr(materializer, "_copy_documents", mutate_validation)
    with pytest.raises(
        (DataAuditError, DataSourceError),
        match=r"SHA256 mismatch|validation|size-mismatched",
    ):
        _run(fixture)
    assert not Path(fixture.arguments["output_root"]).exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"required_source_tokens": {"shared_source": 21}},
            "required source capacity",
        ),
        (
            {"required_aggregate_tokens": 21},
            "aggregate materialization capacity",
        ),
        (
            {"formal_training_tokens": 20, "global_batch_tokens": 1},
            "one-global-batch capacity",
        ),
        (
            {"required_source_tokens": {}},
            "per-source capacity pins differ",
        ),
    ],
)
def test_capacity_contract_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(DataAuditError, match=message):
        _run(fixture, **overrides)


def test_existing_output_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = Path(fixture.arguments["output_root"])
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(DataAuditError, match="already exists"):
        _run(fixture)
    assert sentinel.read_text(encoding="utf-8") == "keep"
