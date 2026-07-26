from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "locks/base-data-sources-v4.json"
DOC_PATH = ROOT / "docs/V4_DATA_SOURCES.zh-CN.md"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_PROFILE_TOKENS = {
    "smoke": 20_000_000,
    "pilot": 250_000_000,
    "dense": 500_000_000,
}

EXPECTED_LOCKED_FILES = {
    "english_fineweb_edu_dedup": (
        "fineweb-edu-dedup/train-00068-of-00234.parquet",
        2_251_835_383,
        "245efe3d53a22ca1d491bfae10478b8a3ae4170df4b89108a3728f6336e34425",
    ),
    "chinese_fineweb2_cmn_hani": (
        "data/cmn_Hani/train/004_00073.parquet",
        1_137_716_850,
        "431fbdf9fd4f8fb807aa7be056cfbf9c57016e0e33d8f73f465703e15b4b5537",
    ),
    "math_finemath_4plus": (
        "finemath-4plus/train-00060-of-00064.parquet",
        284_239_808,
        "d8ee743cf9f6eaa336b99c50b556c6aea1fa34a8209bbd3220716444485aa27f",
    ),
    "code_github_clean_allowlisted": (
        "data/train-00570-of-00880.parquet",
        345_453_831,
        "3df56ea64531cf25a2721320bab619aedf3ca4f4f427bbbe68423d9ecb7c6e15",
    ),
    "science_cosmopedia_openstax": (
        "data/openstax/train-00001-of-00002.parquet",
        173_491_421,
        "05ed6afca68a93f1ec1f062165a49c08dd6d29b2b728165fce7ab96a7f702e24",
    ),
    "science_cosmopedia_stanford": (
        "data/stanford/train-00002-of-00013.parquet",
        253_591_785,
        "a34d00e3b3b130a92f0e637fc2d125c18ac68ae0b480d074691f4b0e6ae6a68f",
    ),
    "education_libretexts_permissive": (
        "libretexts-0000.json.gz",
        115_037_937,
        "71851a2b5acfccf80659729a0eccd07ab6ed737f0edc659a662aa9c1cd2194db",
    ),
    "public_domain_usgpo": (
        "usgpo-0000.json.gz",
        760_402_814,
        "4b99b5043d40e205e578b26e0827f86541bd86f4f11bd04868f61e4259fe4c20",
    ),
    "public_domain_project_gutenberg": (
        "project_gutenberg-dolma-0000.json.gz",
        570_083_371,
        "f7e3716e1e2be607a044920ffec69d1395598cca178d4ccbbd343f206b31effb",
    ),
    "science_arxiv_open_permissive": (
        "arxiv-papers-0007.json.gz",
        210_893_770,
        "3c930859c88959b37519b77f597da9ccbb5bb2825bba1458d216f9ec5c94a90c",
    ),
    "code_stackv2_edu_permissive": (
        "stack-edu-0094.json.gz",
        474_450_587,
        "fab627a4e8add9478c5c08107cf5f2385bab1c3f741a3771a855cb1c32d09f56",
    ),
    "multilingual_common_corpus_permissive": (
        "common_corpus_1/subset_100_1.parquet",
        429_962_586,
        "4ee719a130b7f86978b08c20cc9f490309e2bae2a5c0df4f04fd427365530f07",
    ),
}

EXPECTED_NEW_REVISIONS = {
    "education_libretexts_permissive": (
        "common-pile/libretexts_filtered",
        "70388bca52b4a93515e14b1d56618fd7944988fd",
    ),
    "public_domain_usgpo": (
        "common-pile/usgpo_filtered",
        "b150cc22211de4d57f1b7f570097a00e65042424",
    ),
    "public_domain_project_gutenberg": (
        "common-pile/project_gutenberg_filtered",
        "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7",
    ),
    "science_arxiv_open_permissive": (
        "common-pile/arxiv_papers_filtered",
        "033cf7f53f9b348deec868c1a5a48484f3ee9e52",
    ),
    "code_stackv2_edu_permissive": (
        "common-pile/stackv2_edu_filtered",
        "c354dbe88469a1153e97c6a63ac50591849654de",
    ),
    "multilingual_common_corpus_permissive": (
        "PleIAs/common_corpus",
        "307910e4c5d040d6f318e6edf2a2b97849155771",
    ),
}


def _recipe() -> dict[str, object]:
    with RECIPE_PATH.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _sources_by_id(recipe: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_sources = recipe["sources"]
    assert isinstance(raw_sources, list)
    sources = {
        str(source["source_id"]): source
        for source in raw_sources
        if isinstance(source, dict)
    }
    assert len(sources) == len(raw_sources)
    return sources


def test_v4_recipe_is_an_explicit_runtime_verified_schema_v2_recipe() -> None:
    recipe = _recipe()

    assert recipe["schema_version"] == 2
    assert recipe["schema_status"] == "stable"
    assert recipe["kind"] == "twen_base_data_source_recipe_v2"
    activation = recipe["activation"]
    assert isinstance(activation, dict)
    assert activation["runnable"] is True
    assert activation["current_parser_compatible"] is True
    assert "immutable jsonl_gzip download with LFS SHA256 verification" in activation[
        "required_implementation"
    ]

    formats = recipe["storage_format_contract"]
    assert isinstance(formats, dict)
    assert formats["parquet"]["current_v1_parser_supported"] is True
    assert formats["jsonl_gzip"]["current_v1_parser_supported"] is False
    assert DOC_PATH.is_file()


def test_v4_profile_and_origin_quotas_are_exact() -> None:
    recipe = _recipe()
    sources = _sources_by_id(recipe)
    profiles = recipe["profiles"]
    assert isinstance(profiles, dict)

    assert {
        name: profile["train_tokens"] for name, profile in profiles.items()
    } == EXPECTED_PROFILE_TOKENS

    for profile_name, expected_tokens in EXPECTED_PROFILE_TOKENS.items():
        quota_sum = sum(
            int(source["train_token_quotas"][profile_name])
            for source in sources.values()
        )
        assert quota_sum == expected_tokens

        for source in sources.values():
            basis_points = int(source["mix_basis_points"])
            numerator = expected_tokens * basis_points
            assert numerator % 10_000 == 0
            assert source["train_token_quotas"][profile_name] == numerator // 10_000

    assert sum(int(source["mix_basis_points"]) for source in sources.values()) == 10_000
    assert (
        sum(
            int(source["mix_basis_points"])
            for source in sources.values()
            if source["origin_group"] == "existing"
        )
        == 7_000
    )
    assert (
        sum(
            int(source["mix_basis_points"])
            for source in sources.values()
            if source["origin_group"] == "new"
        )
        == 3_000
    )
    assert (
        sum(int(source["validation_token_quota"]) for source in sources.values())
        == recipe["validation_tokens"]
        == 2_000_000
    )
    for source in sources.values():
        assert int(source["validation_token_quota"]) == (
            recipe["validation_tokens"] * int(source["mix_basis_points"]) // 10_000
        )


def test_v4_sources_use_only_audited_immutable_minimum_files() -> None:
    recipe = _recipe()
    sources = _sources_by_id(recipe)

    assert set(sources) == set(EXPECTED_LOCKED_FILES)
    for source_id, source in sources.items():
        assert HEX40.fullmatch(str(source["revision"]))
        assert source["gated"] is False
        assert source["trust_remote_code"] is False
        assert source["split"] == "train"
        assert source["config"]
        assert source["repo_id"]
        assert str(source["revision"]) in str(source["card_url"])

        locked_files = source["locked_files"]
        assert isinstance(locked_files, list)
        assert len(locked_files) == 1
        locked = locked_files[0]
        expected_path, expected_size, expected_sha256 = EXPECTED_LOCKED_FILES[source_id]
        assert locked == {
            "path": expected_path,
            "size": expected_size,
            "sha256": expected_sha256,
        }
        assert source["file_patterns"] == [expected_path]
        assert HEX64.fullmatch(str(locked["sha256"]))

        storage_format = source["storage_format"]
        if storage_format == "parquet":
            assert expected_path.endswith(".parquet")
        else:
            assert storage_format == "jsonl_gzip"
            assert expected_path.endswith(".json.gz")

        required = set(source["required_fields"])
        assert source["text_field"] in required
        assert set(source["stable_id_fields"]) <= required
        assert source["attribution_fields"]

    for source_id, (repo_id, revision) in EXPECTED_NEW_REVISIONS.items():
        assert sources[source_id]["origin_group"] == "new"
        assert sources[source_id]["repo_id"] == repo_id
        assert sources[source_id]["revision"] == revision


def test_v4_per_document_licenses_are_permissive_and_attributed() -> None:
    recipe = _recipe()
    sources = _sources_by_id(recipe)
    forbidden_fragments = (
        "cc-by-sa",
        "gfdl",
        "gpl",
        "agpl",
        "lgpl",
        "mpl",
        "unknown",
        "no-license",
    )

    for source in sources.values():
        license_field = source.get("license_field")
        if license_field is None:
            continue
        assert license_field in source["required_fields"]
        allowlist = source["license_allowlist"]
        assert allowlist
        for normalized_license in allowlist:
            lowered = normalized_license.lower()
            assert not any(fragment in lowered for fragment in forbidden_fragments)
        assert license_field in source["attribution_fields"]

    assert not any(
        source["repo_id"] == "wikimedia/wikipedia" for source in sources.values()
    )
    common = sources["multilingual_common_corpus_permissive"]
    assert common["row_filters"] == {
        "language_in": ["Chinese", "English"],
        "language_type_in": ["Written"],
        "open_type_in": ["Open Government", "Open Science", "Open Culture"],
        "collection_not_in": ["Wikipedia", "Github Open Source", "Wikidata"],
    }


def test_v4_jsonl_gzip_sources_are_data_only_but_require_the_v2_reader() -> None:
    recipe = _recipe()
    sources = _sources_by_id(recipe)
    gzip_source_ids = {
        source_id
        for source_id, source in sources.items()
        if source["storage_format"] == "jsonl_gzip"
    }

    assert gzip_source_ids == {
        "education_libretexts_permissive",
        "public_domain_usgpo",
        "public_domain_project_gutenberg",
        "science_arxiv_open_permissive",
        "code_stackv2_edu_permissive",
    }
    assert all(sources[source_id]["trust_remote_code"] is False for source_id in gzip_source_ids)
    assert recipe["storage_format_contract"]["jsonl_gzip"][
        "forbidden_shortcut"
    ].startswith("Do not use mutable datasets-server")
