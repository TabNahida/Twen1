"""Authenticated, streaming Base-corpus governance audits.

The audit attestation is deliberately separate from the immutable extractor
manifest.  It binds that manifest, a frozen validation inventory, a benchmark
registry, every findings byte, and the exact scanner source without rewriting
or silently upgrading the original extraction record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
import unicodedata
import urllib.parse
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath

from ..io.download import sha256_file
from ..utils import atomic_write_json, atomic_write_text
from .sources import validate_extracted_base_corpus

AUDIT_SCHEMA_VERSION = 1
AUDIT_KIND = "twen_base_corpus_audit_attestation"
BENCHMARK_REGISTRY_SCHEMA_VERSION = 1
BENCHMARK_REGISTRY_KIND = "twen_benchmark_13gram_registry"
AUDIT_SOURCE_SHA256 = sha256_file(Path(__file__))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEXICAL_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]|[^\w\s]", re.UNICODE)
_EMAIL = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_IPV6 = re.compile(r"(?<![0-9A-F:])(?:[0-9A-F]{1,4}:){2,7}[0-9A-F]{1,4}(?![0-9A-F:])", re.I)
_CONTEXT_PHONE = re.compile(
    r"(?:phone|mobile|telephone|tel|contact|电话|手机|联系方式)\s*[:\uFF1A#-]?\s*"
    r"(?:\+?\d[\d ().-]{6,}\d)",
    re.I,
)
_CONTEXT_ID = re.compile(
    r"(?:ssn|social security|passport|national id|身份证|护照|证件号)\s*[:\uFF1A#-]?\s*"
    r"[A-Z0-9][A-Z0-9 -]{5,24}",
    re.I,
)
_CONTEXT_ADDRESS = re.compile(
    r"(?:home address|mailing address|residential address|家庭住址|家庭地址|收件地址)"
    r"\s*[:\uFF1A]\s*\S.{4,160}",
    re.I,
)
_PAYMENT_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PAYMENT_CONTEXT = re.compile(r"card|credit|debit|银行卡|信用卡|卡号", re.I)
_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "email",
    "password",
    "passwd",
    "phone",
    "secret",
    "token",
}
_GAMBLING_OR_SEO = re.compile(
    r"(?:博彩|赌博|賭博|赌场|賭場|现金网|娱乐城|娛樂城|彩票(?:平台|代理|开户)|"
    r"时时彩|六合彩|百家乐|百家樂|老虎机|老虎機|体育投注|體育投注|"
    r"bet365|博狗博彩|博彩通|皇冠(?:备用|備用|娱乐|娛樂)|"
    r"太阳城(?:娱乐|娛樂))",
    re.I,
)
_GAMBLING_CALL_TO_ACTION = re.compile(
    r"(?:开户|開戶|代理|投注|下注|网址|網址|客服|送彩金|送彩金|返利)",
    re.I,
)
_SEO_STITCHING = re.compile(
    r"(?:备用网址|備用網址|官网入口|官網入口|点击进入|點擊進入|"
    r"关键词优化|關鍵詞優化|SEO优化|SEO優化|"
    r"织梦内容管理系统|織夢內容管理系統|好织梦|dedecms|media_span_url)",
    re.I,
)
_MOJIBAKE = re.compile(
    r"(?:\ufffd|锟斤拷|烫烫烫|屯屯屯|Ã[\x80-\xbf]|Â[\x80-\xbf]|â€[™œ“”])"
)
_BOILERPLATE_MARKERS = (
    re.compile(r"(?:上一篇|上一頁|下一篇|下一頁)\s*[:\N{FULLWIDTH COLON}]?"),
    re.compile(r"(?:相关阅读|相關閱讀|相关文章|相關文章|相关推荐|相關推薦)"),
    re.compile(r"(?:猜你喜欢|猜你喜歡|热门跟帖|熱門跟帖|查看更多评论|查看更多評論)"),
    re.compile(r"(?:扫码(?:手机)?观看|掃碼(?:手機)?觀看|微信[“\"]?扫一扫|微信[「\"]?掃一掃)"),
    re.compile(r"(?:版权与免责声明|版權與免責聲明|免责声明|免責聲明)"),
    re.compile(
        r"(?:责任编辑|責任編輯|浏览数|瀏覽數|点击量|點擊量)"
        r"\s*[:\N{FULLWIDTH COLON}]?"
    ),
    re.compile(r"(?:ICP(?:备|備)案号|公网安备|公網安備)"),
)
# These are deliberately small sets of high-frequency one-to-one variants.
# They are not a language detector.  Rejection requires substantial evidence
# from *both* sets and frequent switching, so normal all-simplified or
# all-traditional documents are not penalized.
_SIMPLIFIED_VARIANTS = frozenset(
    "万与业东严为义乌乐书买云亚产亩亲亿仅从仓仪价众优会体余来侣侧侦"
    "儿党兰关兴养兽冈册写军农冲决况冻净准凉减几凤凯击刘则刚创别删"
    "动务势区医华协单卖卫厂历压厕县发变叶号后吓吗吕吴员听启园围图"
    "国圣场坏块坚坛坝声处备复头夸夹夺奋奖妇妈孙学实审宪宫宽宾寻导"
    "层届岁归当录彻忆忧怀态总惊户扑执扩扫扬扰护报担拟拢拣挂挡挤挥"
    "损换据摆数斗断无时术机权条来极标样树桥梦检楼欢欧歼毁毕气汇汉"
    "没测济爱现电画疗监盘礼离种积称稳笔简签类粮纪约级练组经绝统继"
    "续网罗罚联聪肃胜节范荐药获营虑虚虫补见观规觉触订认让证评识诉"
    "词试话说读谁调谈谊谋谱负财责贤质账货资转轮软达过运还进远选递"
    "逻邮邻郑酿里鉴针钟铁长门间队阳阴陈际随隐难雾静顶项顺须领风飞"
    "马验鱼鸟鸡麦黄齐齿龙"
)
_TRADITIONAL_VARIANTS = frozenset(
    "萬與業東嚴為義烏樂書買雲亞產畝親億僅從倉儀價眾優會體餘來侶側偵"
    "兒黨蘭關興養獸岡冊寫軍農衝決況凍淨準涼減幾鳳凱擊劉則剛創別刪"
    "動務勢區醫華協單賣衛廠歷壓廁縣發變葉號後嚇嗎呂吳員聽啟園圍圖"
    "國聖場壞塊堅壇壩聲處備復頭誇夾奪奮獎婦媽孫學實審憲宮寬賓尋導"
    "層屆歲歸當錄徹憶憂懷態總驚戶撲執擴掃揚擾護報擔擬攏揀掛擋擠揮"
    "損換據擺數鬥斷無時術機權條來極標樣樹橋夢檢樓歡歐殲毀畢氣匯漢"
    "沒測濟愛現電畫療監盤禮離種積稱穩筆簡簽類糧紀約級練組經絕統繼"
    "續網羅罰聯聰肅勝節範薦藥獲營慮虛蟲補見觀規覺觸訂認讓證評識訴"
    "詞試話說讀誰調談誼謀譜負財責賢質賬貨資轉輪軟達過運還進遠選遞"
    "邏郵鄰鄭釀裡鑒針鐘鐵長門間隊陽陰陳際隨隱難霧靜頂項順須領風飛"
    "馬驗魚鳥雞麥黃齊齒龍"
)
_MASK64 = (1 << 64) - 1
_SIGNATURE_BINS = 64
_BAND_SIZE = 4
_SHINGLE_SIZE = 5


class DataAuditError(ValueError):
    """Raised when an audit input or attestation is unauthenticated."""


def content_quality_rejection_reasons(
    text: str,
    *,
    category: str = "general",
) -> tuple[str, ...]:
    """Return deterministic, conservative corpus-quality rejection reasons.

    The policy intentionally records reason codes instead of rewriting text.
    Code is excluded because repeated lines, symbols, and short boundaries are
    often meaningful there.  The caller can therefore count each failure mode
    and project the complete rejection ledger without retaining raw findings.
    """

    if not isinstance(text, str) or category == "code" or "code" in category.casefold():
        return ()
    reasons: list[str] = []
    gambling_hits = tuple(_GAMBLING_OR_SEO.finditer(text))
    seo_stitching_hits = tuple(_SEO_STITCHING.finditer(text))
    if len(gambling_hits) >= 2 or (
        gambling_hits and _GAMBLING_CALL_TO_ACTION.search(text) is not None
    ) or len(seo_stitching_hits) >= 2:
        reasons.append("gambling_or_seo_stitching_spam")

    simplified_positions = [
        index for index, character in enumerate(text) if character in _SIMPLIFIED_VARIANTS
    ]
    traditional_positions = [
        index for index, character in enumerate(text) if character in _TRADITIONAL_VARIANTS
    ]
    variant_total = len(simplified_positions) + len(traditional_positions)
    if (
        len(simplified_positions) >= 8
        and len(traditional_positions) >= 8
        and min(len(simplified_positions), len(traditional_positions)) / variant_total >= 0.12
    ):
        # Build the switch count without depending on locale or a conversion
        # library.  Membership is disjoint for the curated variant sets.
        ordered = sorted(
            [(index, "s") for index in simplified_positions]
            + [(index, "t") for index in traditional_positions]
        )
        switches = sum(left[1] != right[1] for left, right in pairwise(ordered))
        if switches >= 6:
            reasons.append("mixed_chinese_script_conversion_artifact")

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n|\n", text)
    ]
    substantial = [paragraph for paragraph in paragraphs if len(paragraph) >= 48]
    if len(substantial) != len(set(substantial)):
        reasons.append("repeated_paragraph")

    control_count = sum(
        unicodedata.category(character) in {"Cc", "Cf", "Co", "Cs"}
        and character not in "\n\r\t"
        for character in text
    )
    if _MOJIBAKE.search(text) is not None or (
        len(text) >= 200 and control_count / len(text) >= 0.01
    ):
        reasons.append("mojibake_or_garbled_text")

    marker_count = sum(pattern.search(text) is not None for pattern in _BOILERPLATE_MARKERS)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    tiny_line_ratio = (
        sum(len(line) <= 4 for line in nonempty_lines) / len(nonempty_lines)
        if nonempty_lines
        else 0.0
    )
    if marker_count >= 4 or (
        len(nonempty_lines) >= 24 and tiny_line_ratio >= 0.55 and marker_count >= 2
    ):
        reasons.append("crawler_boilerplate_or_abnormal_boundaries")
    return tuple(reasons)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _normalized_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise DataAuditError(f"{field} must be a 64-digit SHA256")
    return value.lower()


def _safe_relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataAuditError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() == "." or ".." in path.parts:
        raise DataAuditError(f"unsafe {field}: {value!r}")
    return path.as_posix()


def _file_identity(path: Path, *, relative: str | None = None) -> dict[str, object]:
    return {
        "path": relative if relative is not None else str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True, slots=True)
class _CorpusInput:
    manifest_path: Path
    manifest_sha256: str
    corpus_fingerprint: str
    value: Mapping[str, object]
    source_by_file: Mapping[str, tuple[str, str]]

    def files(self, role: str) -> tuple[tuple[Path, str, str, str], ...]:
        raw = self.value.get(f"{role}_files")
        if not isinstance(raw, list):
            raise DataAuditError(f"extracted corpus has no {role} inventory")
        result: list[tuple[Path, str, str, str]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise DataAuditError(f"invalid {role} file entry {index}")
            relative = _safe_relative(item.get("path"), f"{role}_files[{index}].path")
            source_id, category = self.source_by_file.get(
                relative,
                (
                    PurePosixPath(relative).parts[-3]
                    if len(PurePosixPath(relative).parts) >= 3
                    else "unknown",
                    "unknown",
                ),
            )
            result.append((self.manifest_path.parent / relative, relative, source_id, category))
        return tuple(result)


def _load_corpus(path: str | Path) -> _CorpusInput:
    manifest = Path(path).resolve()
    # This authenticates COMPLETE, the manifest fingerprint, sidecars, and all
    # referenced JSONL/ledger size+SHA values before any scanner trusts them.
    validate_extracted_base_corpus(manifest, verify_hashes=True)
    raw = manifest.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping):  # pragma: no cover - validator already checked
        raise DataAuditError("extracted manifest must be an object")
    mapping: dict[str, tuple[str, str]] = {}
    sources = value.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            source_id = str(source.get("source_id", "unknown"))
            category = str(source.get("category", "unknown"))
            chunks = source.get("chunks")
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, Mapping) or not isinstance(chunk.get("outputs"), list):
                    continue
                for output in chunk["outputs"]:
                    if isinstance(output, Mapping) and isinstance(output.get("path"), str):
                        mapping[str(output["path"])] = (source_id, category)
    return _CorpusInput(
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        corpus_fingerprint=_normalized_sha(value.get("corpus_fingerprint"), "corpus_fingerprint"),
        value=value,
        source_by_file=mapping,
    )


def _iter_jsonl_documents(
    corpus: _CorpusInput,
    role: str,
) -> Iterator[tuple[str, str, str, int, str]]:
    for path, relative, source_id, category in corpus.files(role):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataAuditError(f"invalid JSONL at {path}:{line_number}") from error
                text = value.get("text") if isinstance(value, Mapping) else None
                if not isinstance(text, str) or not text.strip():
                    raise DataAuditError(f"missing text at {path}:{line_number}")
                yield relative, source_id, category, line_number, text


def _normalize_text(text: str, *, code: bool) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not code:
        normalized = unicodedata.normalize("NFKC", " ".join(normalized.split()))
    return normalized


def _lexical_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(match.group(0) for match in _LEXICAL_TOKEN.finditer(normalized))


def _one_permutation_signature(tokens: Sequence[str], content_sha: str) -> tuple[int, ...]:
    values = [zlib.crc32(token.encode("utf-8")) for token in tokens]
    signature = [_MASK64] * _SIGNATURE_BINS
    if len(values) >= _SHINGLE_SIZE:
        multiplier = 0x9E3779B185EBCA87
        power = pow(multiplier, _SHINGLE_SIZE - 1, 1 << 64)
        rolling = 0
        for value in values[:_SHINGLE_SIZE]:
            rolling = (rolling * multiplier + value + 1) & _MASK64
        for index in range(len(values) - _SHINGLE_SIZE + 1):
            if index:
                outgoing = values[index - 1] + 1
                incoming = values[index + _SHINGLE_SIZE - 1] + 1
                rolling = ((rolling - outgoing * power) * multiplier + incoming) & _MASK64
            mixed = (rolling ^ (rolling >> 33)) * 0xFF51AFD7ED558CCD & _MASK64
            mixed = (mixed ^ (mixed >> 33)) * 0xC4CEB9FE1A85EC53 & _MASK64
            mixed ^= mixed >> 33
            bucket = mixed % _SIGNATURE_BINS
            value = mixed // _SIGNATURE_BINS
            if value < signature[bucket]:
                signature[bucket] = value
    occupied = tuple(index for index, value in enumerate(signature) if value != _MASK64)
    for bucket, value in enumerate(signature):
        if value != _MASK64:
            continue
        if occupied:
            distance = next(
                offset
                for offset in range(1, _SIGNATURE_BINS + 1)
                if (bucket + offset) % _SIGNATURE_BINS in occupied
            )
            source = (bucket + distance) % _SIGNATURE_BINS
            salt = int.from_bytes(
                hashlib.sha256(f"densify\0{bucket}\0{distance}".encode()).digest()[:8],
                "big",
            )
            signature[bucket] = signature[source] ^ salt
        else:
            signature[bucket] = int.from_bytes(
                hashlib.sha256(f"empty\0{bucket}\0{content_sha}".encode()).digest()[:8],
                "big",
            )
    return tuple(signature)


def _packed_signature(signature: Sequence[int]) -> bytes:
    if len(signature) != _SIGNATURE_BINS:
        raise DataAuditError("invalid near-dedup signature length")
    return struct.pack(f">{_SIGNATURE_BINS}Q", *signature)


def _unpacked_signature(value: bytes) -> tuple[int, ...]:
    return struct.unpack(f">{_SIGNATURE_BINS}Q", value)


def _band_keys(signature: Sequence[int]) -> tuple[bytes, ...]:
    packed = _packed_signature(signature)
    band_bytes = _BAND_SIZE * 8
    return tuple(
        bytes([band // band_bytes]) + packed[band : band + band_bytes]
        for band in range(0, len(packed), band_bytes)
    )


def _signature_similarity(first: Sequence[int], second: Sequence[int]) -> float:
    return sum(left == right for left, right in zip(first, second, strict=True)) / len(first)


def _luhn_valid(candidate: str) -> bool:
    digits = [int(value) for value in candidate if value.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _contextual_pii_categories(text: str) -> tuple[str, ...]:
    categories: set[str] = set()
    if _EMAIL.search(text):
        categories.add("email")
    for match in _IPV4.finditer(text):
        if all(0 <= int(part) <= 255 for part in match.group(0).split(".")):
            categories.add("ipv4")
            break
    if _IPV6.search(text):
        categories.add("ipv6")
    if _CONTEXT_PHONE.search(text):
        categories.add("contextual_phone")
    if _CONTEXT_ID.search(text):
        categories.add("contextual_government_id")
    if _CONTEXT_ADDRESS.search(text):
        categories.add("contextual_address")
    if _PAYMENT_CONTEXT.search(text) and any(
        _luhn_valid(match.group(0)) for match in _PAYMENT_CANDIDATE.finditer(text)
    ):
        categories.add("payment_card_luhn")
    for raw_url in re.findall(r"https?://[^\s<>\"']+", text, flags=re.I):
        try:
            keys = {
                key.casefold()
                for key, _ in urllib.parse.parse_qsl(urllib.parse.urlsplit(raw_url).query)
            }
        except ValueError:
            continue
        if keys & _SENSITIVE_QUERY_KEYS:
            categories.add("sensitive_url_query")
            break
    return tuple(sorted(categories))


def _text_13gram_hashes(text: str) -> Iterator[str]:
    tokens = _lexical_tokens(text)
    for index in range(max(0, len(tokens) - 12)):
        yield hashlib.sha256("\0".join(tokens[index : index + 13]).encode()).hexdigest()


def inspect_benchmark_registry(
    registry_path: str | Path,
    *,
    benchmark_root: str | Path,
    verify_hashes: bool = True,
) -> dict[str, object]:
    path = Path(registry_path).resolve()
    root = Path(benchmark_root).resolve()
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataAuditError(f"invalid benchmark registry JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise DataAuditError("benchmark registry must be an object")
    if value.get("schema_version") != BENCHMARK_REGISTRY_SCHEMA_VERSION:
        raise DataAuditError("unsupported benchmark registry schema")
    if value.get("kind") != BENCHMARK_REGISTRY_KIND:
        raise DataAuditError("unexpected benchmark registry kind")
    raw_benchmarks = value.get("benchmarks")
    if not isinstance(raw_benchmarks, list) or not raw_benchmarks:
        raise DataAuditError("benchmark registry must contain benchmarks")
    pending: list[str] = []
    ready: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw_benchmark in enumerate(raw_benchmarks):
        if not isinstance(raw_benchmark, Mapping):
            raise DataAuditError(f"benchmarks[{index}] must be an object")
        benchmark_id = raw_benchmark.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not benchmark_id or benchmark_id in seen_ids:
            raise DataAuditError(f"invalid/duplicate benchmark_id: {benchmark_id!r}")
        seen_ids.add(benchmark_id)
        required = raw_benchmark.get("required", True)
        if not isinstance(required, bool):
            raise DataAuditError(f"{benchmark_id}.required must be boolean")
        status = raw_benchmark.get("status")
        if status != "ready":
            if required:
                pending.append(benchmark_id)
            continue
        revision = raw_benchmark.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            raise DataAuditError(f"ready benchmark {benchmark_id} needs an immutable revision")
        raw_files = raw_benchmark.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise DataAuditError(f"ready benchmark {benchmark_id} has no files")
        files: list[dict[str, object]] = []
        for file_index, raw_file in enumerate(raw_files):
            if not isinstance(raw_file, Mapping):
                raise DataAuditError(f"{benchmark_id}.files[{file_index}] must be an object")
            relative = _safe_relative(
                raw_file.get("path"), f"{benchmark_id}.files[{file_index}].path"
            )
            file_path = (root / relative).resolve()
            try:
                file_path.relative_to(root)
            except ValueError as error:
                raise DataAuditError(f"benchmark file escapes root: {relative}") from error
            expected_size = raw_file.get("size")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                raise DataAuditError(f"invalid benchmark file size: {relative}")
            expected_sha = _normalized_sha(raw_file.get("sha256"), f"{relative}.sha256")
            fields = raw_file.get("text_fields")
            if (
                not isinstance(fields, list)
                or not fields
                or not all(isinstance(item, str) and item for item in fields)
            ):
                raise DataAuditError(f"{relative}.text_fields must be non-empty strings")
            if raw_file.get("format") != "jsonl":
                raise DataAuditError(f"{relative}.format must currently be jsonl")
            if not file_path.is_file() or file_path.stat().st_size != expected_size:
                raise DataAuditError(f"missing/size-mismatched benchmark file: {file_path}")
            if verify_hashes and sha256_file(file_path) != expected_sha:
                raise DataAuditError(f"benchmark SHA256 mismatch: {file_path}")
            files.append(
                {
                    "path": relative,
                    "absolute_path": str(file_path),
                    "size": expected_size,
                    "sha256": expected_sha,
                    "format": "jsonl",
                    "text_fields": list(fields),
                }
            )
        ready.append({"benchmark_id": benchmark_id, "files": files})
    return {
        "ok": True,
        "registry_path": str(path),
        "registry_sha256": hashlib.sha256(raw).hexdigest(),
        "benchmark_root": str(root),
        "ready": not pending,
        "pending_benchmarks": sorted(pending),
        "ready_benchmarks": ready,
    }


def _field_value(record: Mapping[str, object], dotted: str) -> object:
    value: object = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _iter_benchmark_texts(report: Mapping[str, object]) -> Iterator[tuple[str, str]]:
    benchmarks = report.get("ready_benchmarks")
    if not isinstance(benchmarks, list):
        return
    for benchmark in benchmarks:
        if not isinstance(benchmark, Mapping):
            continue
        benchmark_id = str(benchmark["benchmark_id"])
        files = benchmark.get("files")
        if not isinstance(files, list):
            continue
        for file in files:
            if not isinstance(file, Mapping):
                continue
            fields = tuple(str(value) for value in file["text_fields"])
            with Path(str(file["absolute_path"])).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise DataAuditError(
                            f"invalid benchmark JSONL at {file['absolute_path']}:{line_number}"
                        ) from error
                    if not isinstance(record, Mapping):
                        raise DataAuditError("benchmark JSONL rows must be objects")
                    for field in fields:
                        value = _field_value(record, field)
                        values = value if isinstance(value, list) else [value]
                        for item in values:
                            if isinstance(item, str) and item.strip():
                                yield benchmark_id, item


def _create_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE docs (
            id INTEGER PRIMARY KEY,
            content_sha TEXT NOT NULL,
            source_id TEXT NOT NULL,
            role TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            signature BLOB NOT NULL
        );
        CREATE INDEX docs_content_sha ON docs(content_sha);
        CREATE TABLE bands (band_key BLOB NOT NULL, doc_id INTEGER NOT NULL);
        CREATE INDEX bands_key ON bands(band_key);
        """
    )
    return connection


def _finding_document(
    *, role: str, source_id: str, path: str, line_number: int, content_sha: str
) -> dict[str, object]:
    return {
        "role": role,
        "source_id": source_id,
        "path": path,
        "line": line_number,
        "text_sha256": content_sha,
    }


def build_base_audit_attestation(
    extracted_manifest: str | Path,
    frozen_validation_manifest: str | Path,
    registry_path: str | Path,
    benchmark_root: str | Path,
    output_root: str | Path,
    *,
    near_duplicate_threshold: float = 0.8,
    max_findings: int = 10_000,
) -> Path:
    """Run all v2 gates and atomically emit an authenticated attestation."""

    if not 0.5 <= near_duplicate_threshold <= 1.0:
        raise DataAuditError("near_duplicate_threshold must be in [0.5, 1.0]")
    if isinstance(max_findings, bool) or max_findings <= 0:
        raise DataAuditError("max_findings must be positive")
    candidate = _load_corpus(extracted_manifest)
    frozen = _load_corpus(frozen_validation_manifest)
    registry = inspect_benchmark_registry(
        registry_path,
        benchmark_root=benchmark_root,
        verify_hashes=True,
    )
    root = Path(output_root).resolve()
    if root.exists():
        raise DataAuditError(f"audit output already exists; choose a new directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{root.name}.incomplete-", dir=root.parent))
    findings_path = work / "findings.jsonl"
    rejections_path = work / "rejections.jsonl"
    index_path = work / "near-dedup.sqlite3"
    metrics = {
        "frozen_validation_documents": 0,
        "candidate_train_documents": 0,
        "train_validation_exact_matches": 0,
        "cross_source_exact_matches": 0,
        "train_validation_near_matches": 0,
        "cross_source_near_matches": 0,
        "contextual_pii_documents": 0,
        "content_quality_documents": 0,
        "content_quality_reasons": {
            "gambling_or_seo_stitching_spam": 0,
            "mixed_chinese_script_conversion_artifact": 0,
            "repeated_paragraph": 0,
            "mojibake_or_garbled_text": 0,
            "crawler_boilerplate_or_abnormal_boundaries": 0,
        },
        "benchmark_overlap_documents": 0,
        "rejection_events": 0,
        "findings_recorded": 0,
        "findings_truncated": 0,
    }
    benchmark_ngrams: dict[str, str] = {}
    for benchmark_id, text in _iter_benchmark_texts(registry):
        for digest in _text_13gram_hashes(text):
            benchmark_ngrams.setdefault(digest, benchmark_id)

    connection = _create_index(index_path)
    try:
        with (
            findings_path.open("w", encoding="utf-8") as findings,
            rejections_path.open("w", encoding="utf-8") as rejections,
        ):

            def record(payload: Mapping[str, object]) -> None:
                gate = payload.get("gate")
                document = payload.get("document")
                if isinstance(gate, str) and isinstance(document, Mapping):
                    rejections.write(
                        json.dumps(
                            {"gate": gate, "document": dict(document)},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    metrics["rejection_events"] += 1
                if metrics["findings_recorded"] < max_findings:
                    findings.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    metrics["findings_recorded"] += 1
                else:
                    metrics["findings_truncated"] += 1

            def scan_document(
                *,
                role: str,
                relative: str,
                source_id: str,
                category: str,
                line_number: int,
                text: str,
            ) -> None:
                code = category == "code" or "code" in source_id.casefold()
                normalized = _normalize_text(text, code=code)
                content_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                document = _finding_document(
                    role=role,
                    source_id=source_id,
                    path=relative,
                    line_number=line_number,
                    content_sha=content_sha,
                )
                quality_reasons = content_quality_rejection_reasons(
                    text,
                    category=category,
                )
                if quality_reasons:
                    metrics["content_quality_documents"] += 1
                    reason_metrics = metrics["content_quality_reasons"]
                    assert isinstance(reason_metrics, dict)
                    for reason in quality_reasons:
                        reason_metrics[reason] += 1
                    record(
                        {
                            "gate": "deterministic_content_quality_scan",
                            "document": document,
                            "reasons": list(quality_reasons),
                        }
                    )
                categories = _contextual_pii_categories(text)
                if categories:
                    metrics["contextual_pii_documents"] += 1
                    record(
                        {
                            "gate": "full_contextual_pii_scan",
                            "document": document,
                            "categories": list(categories),
                        }
                    )

                if role == "train" and benchmark_ngrams:
                    match = next(
                        (
                            (digest, benchmark_ngrams[digest])
                            for digest in _text_13gram_hashes(text)
                            if digest in benchmark_ngrams
                        ),
                        None,
                    )
                    if match is not None:
                        metrics["benchmark_overlap_documents"] += 1
                        record(
                            {
                                "gate": "project_benchmark_13gram_scan",
                                "document": document,
                                "benchmark_id": match[1],
                                "ngram_sha256": match[0],
                            }
                        )

                validation_exact = None
                if role == "train":
                    validation_exact = connection.execute(
                        "SELECT source_id, file_path, line_number FROM docs "
                        "WHERE content_sha=? AND role='validation' LIMIT 1",
                        (content_sha,),
                    ).fetchone()
                    if validation_exact is not None:
                        metrics["train_validation_exact_matches"] += 1
                        record(
                            {
                                "gate": "train_vs_frozen_validation_exact_dedup",
                                "document": document,
                                "matched_document": {
                                    "role": "validation",
                                    "source_id": validation_exact[0],
                                    "path": validation_exact[1],
                                    "line": validation_exact[2],
                                },
                            }
                        )
                cross_exact = connection.execute(
                    "SELECT role, source_id, file_path, line_number FROM docs "
                    "WHERE content_sha=? AND source_id<>? LIMIT 1",
                    (content_sha, source_id),
                ).fetchone()
                if cross_exact is not None:
                    metrics["cross_source_exact_matches"] += 1
                    record(
                        {
                            "gate": "cross_source_exact_dedup",
                            "document": document,
                            "matched_document": {
                                "role": cross_exact[0],
                                "source_id": cross_exact[1],
                                "path": cross_exact[2],
                                "line": cross_exact[3],
                            },
                        }
                    )

                tokens = _lexical_tokens(normalized)
                signature = _one_permutation_signature(tokens, content_sha)
                candidate_ids: set[int] = set()
                for key in _band_keys(signature):
                    candidate_ids.update(
                        int(row[0])
                        for row in connection.execute(
                            "SELECT doc_id FROM bands WHERE band_key=?", (key,)
                        )
                    )
                validation_near = None
                cross_near = None
                for doc_id in candidate_ids:
                    row = connection.execute(
                        "SELECT content_sha, source_id, role, file_path, line_number, signature "
                        "FROM docs WHERE id=?",
                        (doc_id,),
                    ).fetchone()
                    if row is None or row[0] == content_sha:
                        continue
                    similarity = _signature_similarity(signature, _unpacked_signature(row[5]))
                    if similarity < near_duplicate_threshold:
                        continue
                    if role == "train" and row[2] == "validation" and validation_near is None:
                        validation_near = (*row[1:5], similarity)
                    if row[1] != source_id and cross_near is None:
                        cross_near = (*row[1:5], similarity)
                if validation_near is not None:
                    metrics["train_validation_near_matches"] += 1
                    record(
                        {
                            "gate": "train_vs_frozen_validation_near_dedup",
                            "document": document,
                            "matched_document": {
                                "source_id": validation_near[0],
                                "role": validation_near[1],
                                "path": validation_near[2],
                                "line": validation_near[3],
                            },
                            "estimated_jaccard": validation_near[4],
                        }
                    )
                if cross_near is not None:
                    metrics["cross_source_near_matches"] += 1
                    record(
                        {
                            "gate": "cross_source_near_dedup",
                            "document": document,
                            "matched_document": {
                                "source_id": cross_near[0],
                                "role": cross_near[1],
                                "path": cross_near[2],
                                "line": cross_near[3],
                            },
                            "estimated_jaccard": cross_near[4],
                        }
                    )

                packed = _packed_signature(signature)
                cursor = connection.execute(
                    "INSERT INTO docs(content_sha,source_id,role,file_path,line_number,signature) "
                    "VALUES(?,?,?,?,?,?)",
                    (content_sha, source_id, role, relative, line_number, packed),
                )
                doc_id = int(cursor.lastrowid)
                connection.executemany(
                    "INSERT INTO bands(band_key,doc_id) VALUES(?,?)",
                    ((key, doc_id) for key in _band_keys(signature)),
                )

            for relative, source_id, category, line_number, text in _iter_jsonl_documents(
                frozen, "validation"
            ):
                scan_document(
                    role="validation",
                    relative=relative,
                    source_id=source_id,
                    category=category,
                    line_number=line_number,
                    text=text,
                )
                metrics["frozen_validation_documents"] += 1
                if metrics["frozen_validation_documents"] % 1000 == 0:
                    connection.commit()
            connection.commit()
            for relative, source_id, category, line_number, text in _iter_jsonl_documents(
                candidate, "train"
            ):
                scan_document(
                    role="train",
                    relative=relative,
                    source_id=source_id,
                    category=category,
                    line_number=line_number,
                    text=text,
                )
                metrics["candidate_train_documents"] += 1
                if metrics["candidate_train_documents"] % 1000 == 0:
                    connection.commit()
            connection.commit()
    finally:
        connection.close()

    benchmark_ready = bool(registry["ready"])
    gates = {
        "train_vs_frozen_validation_exact_dedup": {
            "status": (
                "complete_no_matches"
                if metrics["train_validation_exact_matches"] == 0
                else "failed_matches_found"
            ),
            "passed": metrics["train_validation_exact_matches"] == 0,
        },
        "cross_source_exact_dedup": {
            "status": (
                "complete_no_matches"
                if metrics["cross_source_exact_matches"] == 0
                else "failed_matches_found"
            ),
            "passed": metrics["cross_source_exact_matches"] == 0,
        },
        "cross_source_near_dedup": {
            "status": (
                "complete_no_matches_minhash_lsh_v1"
                if metrics["cross_source_near_matches"] == 0
                and metrics["train_validation_near_matches"] == 0
                else "failed_matches_found"
            ),
            "passed": metrics["cross_source_near_matches"] == 0
            and metrics["train_validation_near_matches"] == 0,
        },
        "full_contextual_pii_scan": {
            "status": (
                "complete_policy_v1_no_findings"
                if metrics["contextual_pii_documents"] == 0
                else "failed_findings_present"
            ),
            "passed": metrics["contextual_pii_documents"] == 0,
        },
        "deterministic_content_quality_scan": {
            "status": (
                "complete_policy_v1_no_findings"
                if metrics["content_quality_documents"] == 0
                else "failed_findings_present"
            ),
            "passed": metrics["content_quality_documents"] == 0,
        },
        "project_benchmark_13gram_scan": {
            "status": (
                "pending_benchmark_registry"
                if not benchmark_ready
                else (
                    "complete_no_overlaps"
                    if metrics["benchmark_overlap_documents"] == 0
                    else "failed_overlaps_found"
                )
            ),
            "passed": benchmark_ready and metrics["benchmark_overlap_documents"] == 0,
        },
    }
    ready_for_training = all(bool(value["passed"]) for value in gates.values())
    findings_identity = _file_identity(findings_path, relative="findings.jsonl")
    rejections_identity = _file_identity(rejections_path, relative="rejections.jsonl")
    candidate_identity = {
        "manifest_path": str(candidate.manifest_path),
        "manifest_sha256": candidate.manifest_sha256,
        "corpus_fingerprint": candidate.corpus_fingerprint,
        "role": "train",
    }
    frozen_identity = {
        "manifest_path": str(frozen.manifest_path),
        "manifest_sha256": frozen.manifest_sha256,
        "corpus_fingerprint": frozen.corpus_fingerprint,
        "role": "validation",
    }
    payload: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "audit_source_sha256": AUDIT_SOURCE_SHA256,
        "candidate": candidate_identity,
        "frozen_validation": frozen_identity,
        "benchmark_registry": {
            "path": registry["registry_path"],
            "sha256": registry["registry_sha256"],
            "benchmark_root": registry["benchmark_root"],
            "ready": benchmark_ready,
            "pending_benchmarks": registry["pending_benchmarks"],
            "ready_benchmarks": registry["ready_benchmarks"],
            "indexed_13grams": len(benchmark_ngrams),
        },
        "policy": {
            "near_duplicate": {
                "algorithm": "lexical-5gram-one-permutation-minhash-lsh-v1",
                "signature_bins": _SIGNATURE_BINS,
                "band_size": _BAND_SIZE,
                "estimated_jaccard_threshold": near_duplicate_threshold,
            },
            "contextual_pii": "deterministic-contextual-regex+luhn+url-query-v1",
            "content_quality": (
                "gambling-seo+mixed-script+repeated-paragraph+mojibake+"
                "crawler-boilerplate-deterministic-v1"
            ),
            "benchmark_overlap": "unicode-nfkc-casefold-lexical-13gram-sha256-v1",
            "findings_store_raw_text": False,
            "max_findings": max_findings,
        },
        "metrics": metrics,
        "gates": gates,
        "findings": findings_identity,
        "rejection_ledger": {
            **rejections_identity,
            "complete": True,
            "stores_raw_text": False,
        },
        "ready_for_training": ready_for_training,
    }
    payload["attestation_fingerprint"] = _canonical_sha256(payload)
    attestation = work / "attestation.json"
    atomic_write_json(attestation, payload)
    atomic_write_json(
        work / "COMPLETE",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "kind": "twen_base_corpus_audit_complete",
            "attestation": attestation.name,
            "attestation_sha256": sha256_file(attestation),
            "attestation_fingerprint": payload["attestation_fingerprint"],
            "ready_for_training": ready_for_training,
        },
    )
    index_path.unlink(missing_ok=True)
    Path(str(index_path) + "-wal").unlink(missing_ok=True)
    Path(str(index_path) + "-shm").unlink(missing_ok=True)
    try:
        os.replace(work, root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return root / "attestation.json"


def validate_base_audit_attestation(path: str | Path) -> Mapping[str, object]:
    attestation = Path(path).resolve()
    raw = attestation.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataAuditError(f"invalid audit attestation: {attestation}") from error
    if not isinstance(value, dict):
        raise DataAuditError("audit attestation must be an object")
    if value.get("schema_version") != AUDIT_SCHEMA_VERSION or value.get("kind") != AUDIT_KIND:
        raise DataAuditError("unsupported audit attestation")
    if value.get("audit_source_sha256") != AUDIT_SOURCE_SHA256:
        raise DataAuditError("audit scanner source changed; rerun the corpus audit")
    fingerprint = _normalized_sha(value.get("attestation_fingerprint"), "attestation_fingerprint")
    fingerprint_payload = dict(value)
    fingerprint_payload.pop("attestation_fingerprint")
    if _canonical_sha256(fingerprint_payload) != fingerprint:
        raise DataAuditError("audit attestation fingerprint mismatch")
    for field in ("candidate", "frozen_validation"):
        identity = value.get(field)
        if not isinstance(identity, Mapping):
            raise DataAuditError(f"audit {field} identity is invalid")
        manifest = Path(str(identity.get("manifest_path"))).resolve()
        if sha256_file(manifest) != _normalized_sha(
            identity.get("manifest_sha256"), f"{field}.manifest_sha256"
        ):
            raise DataAuditError(f"audit {field} manifest changed")
    registry = value.get("benchmark_registry")
    if not isinstance(registry, Mapping):
        raise DataAuditError("audit benchmark registry identity is invalid")
    registry_path = Path(str(registry.get("path"))).resolve()
    if sha256_file(registry_path) != _normalized_sha(
        registry.get("sha256"), "benchmark_registry.sha256"
    ):
        raise DataAuditError("audit benchmark registry changed")
    findings = value.get("findings")
    if not isinstance(findings, Mapping):
        raise DataAuditError("audit findings identity is invalid")
    relative = _safe_relative(findings.get("path"), "findings.path")
    findings_path = attestation.parent / relative
    if not findings_path.is_file() or findings_path.stat().st_size != findings.get("size"):
        raise DataAuditError("audit findings file is missing or size-mismatched")
    if sha256_file(findings_path) != _normalized_sha(findings.get("sha256"), "findings.sha256"):
        raise DataAuditError("audit findings SHA256 mismatch")
    rejection_ledger = value.get("rejection_ledger")
    if not isinstance(rejection_ledger, Mapping) or rejection_ledger.get("complete") is not True:
        raise DataAuditError("audit rejection ledger is missing or incomplete")
    rejection_relative = _safe_relative(rejection_ledger.get("path"), "rejection_ledger.path")
    rejection_path = attestation.parent / rejection_relative
    if not rejection_path.is_file() or rejection_path.stat().st_size != rejection_ledger.get(
        "size"
    ):
        raise DataAuditError("audit rejection ledger is missing or size-mismatched")
    if sha256_file(rejection_path) != _normalized_sha(
        rejection_ledger.get("sha256"), "rejection_ledger.sha256"
    ):
        raise DataAuditError("audit rejection ledger SHA256 mismatch")
    complete_path = attestation.parent / "COMPLETE"
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataAuditError("audit output has no valid COMPLETE marker") from error
    if (
        not isinstance(complete, Mapping)
        or complete.get("kind") != "twen_base_corpus_audit_complete"
        or complete.get("attestation") != attestation.name
        or complete.get("attestation_sha256") != hashlib.sha256(raw).hexdigest()
        or complete.get("attestation_fingerprint") != fingerprint
        or complete.get("ready_for_training") != value.get("ready_for_training")
    ):
        raise DataAuditError("audit COMPLETE metadata mismatch")
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise DataAuditError("audit gates are missing")
    computed_ready = all(
        isinstance(item, Mapping) and item.get("passed") is True for item in gates.values()
    )
    if value.get("ready_for_training") is not computed_ready:
        raise DataAuditError("audit ready_for_training differs from gate results")
    return value


def _load_rejection_ledger(
    attestation_path: Path,
    value: Mapping[str, object],
) -> tuple[set[tuple[str, str, int]], dict[str, int]]:
    identity = value.get("rejection_ledger")
    if not isinstance(identity, Mapping):  # validator gives the user-facing error
        raise DataAuditError("audit rejection ledger is missing")
    path = attestation_path.parent / _safe_relative(identity.get("path"), "rejection_ledger.path")
    keys: set[tuple[str, str, int]] = set()
    reasons: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataAuditError(
                    f"invalid rejection ledger JSONL at {path}:{line_number}"
                ) from error
            if not isinstance(row, Mapping) or not isinstance(row.get("document"), Mapping):
                raise DataAuditError("invalid rejection ledger row")
            gate = row.get("gate")
            document = row["document"]
            role = document.get("role")
            relative = document.get("path")
            source_line = document.get("line")
            if (
                not isinstance(gate, str)
                or role not in {"train", "validation"}
                or not isinstance(relative, str)
                or isinstance(source_line, bool)
                or not isinstance(source_line, int)
                or source_line <= 0
            ):
                raise DataAuditError("invalid rejection ledger document identity")
            keys.add((role, relative, source_line))
            reasons[gate] = reasons.get(gate, 0) + 1
    return keys, reasons


def _filtered_output_name(role: str, source_id: str, index: int) -> str:
    safe_source = re.sub(r"[^A-Za-z0-9._-]+", "-", source_id).strip("-.") or "unknown"
    # Keep the canonical role basename because the authenticated source-map
    # contract derives output ownership from ``train.jsonl`` and
    # ``validation.jsonl`` entries in each source chunk.
    return f"filtered/{safe_source}/chunk-{index:06d}/{role}.jsonl"


def materialize_filtered_base_corpus(
    attestation_path: str | Path,
    output_root: str | Path,
) -> Path:
    """Materialize the complement of the complete rejection ledger.

    The resulting extracted corpus remains pending until ``audit-base`` is run
    again against its own filtered validation inventory.  This two-pass
    contract prevents a sanitizer from certifying its own output without an
    independent rescan.
    """

    attestation = Path(attestation_path).resolve()
    value = validate_base_audit_attestation(attestation)
    candidate_identity = value.get("candidate")
    frozen_identity = value.get("frozen_validation")
    if not isinstance(candidate_identity, Mapping) or not isinstance(frozen_identity, Mapping):
        raise DataAuditError("audit corpus identities are invalid")
    candidate = _load_corpus(str(candidate_identity["manifest_path"]))
    frozen = _load_corpus(str(frozen_identity["manifest_path"]))
    if candidate.manifest_sha256 != candidate_identity.get("manifest_sha256"):
        raise DataAuditError("candidate manifest differs from attestation")
    if frozen.manifest_sha256 != frozen_identity.get("manifest_sha256"):
        raise DataAuditError("frozen validation manifest differs from attestation")
    if candidate.value.get("tokenizer_manifest_sha256") != frozen.value.get(
        "tokenizer_manifest_sha256"
    ):
        raise DataAuditError("candidate and frozen validation tokenizer identities differ")
    rejection_keys, rejection_reasons = _load_rejection_ledger(attestation, value)

    root = Path(output_root).resolve()
    if root.exists():
        raise DataAuditError(f"filtered output already exists; choose a new directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{root.name}.incomplete-", dir=root.parent))
    role_files: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    source_map_outputs: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
    }
    source_outputs: dict[str, dict[str, object]] = {}
    kept_document_keys: set[tuple[str, str, str]] = set()
    role_documents = {"train": 0, "validation": 0}
    source_role_documents: dict[str, dict[str, int]] = {
        "train": {},
        "validation": {},
    }
    rejected_documents = {"train": 0, "validation": 0}
    contract_names = (
        "source_map",
        "source_mix",
        "format_audit",
        "license_audit",
        "materialization_audit",
    )
    contract_presence = [name in candidate.value for name in contract_names]
    if any(contract_presence) and not all(contract_presence):
        # The corpus validator normally catches this first.  Keep the
        # materializer fail-closed if a different authenticated loader is ever
        # introduced.
        raise DataAuditError("candidate corpus has a partial data-contract audit")
    has_data_contract = all(contract_presence)
    try:
        for role, corpus in (("train", candidate), ("validation", frozen)):
            for index, (source, relative, source_id, category) in enumerate(corpus.files(role)):
                output_relative = _filtered_output_name(role, source_id, index)
                output = work / output_relative
                output.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with (
                    source.open("r", encoding="utf-8") as source_handle,
                    output.open("w", encoding="utf-8") as output_handle,
                ):
                    for line_number, line in enumerate(source_handle, start=1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise DataAuditError(
                                f"invalid source JSONL at {source}:{line_number}"
                            ) from error
                        text = row.get("text") if isinstance(row, Mapping) else None
                        if not isinstance(text, str):
                            raise DataAuditError(f"missing source text at {source}:{line_number}")
                        if (role, relative, line_number) in rejection_keys:
                            rejected_documents[role] += 1
                            continue
                        output_handle.write(line if line.endswith("\n") else line + "\n")
                        written += 1
                        content_sha = hashlib.sha256(
                            _normalize_text(
                                text,
                                code=category == "code" or "code" in source_id.casefold(),
                            ).encode("utf-8")
                        ).hexdigest()
                        kept_document_keys.add((role, source_id, content_sha))
                if not written:
                    output.unlink()
                    continue
                role_documents[role] += written
                source_role_documents[role][source_id] = (
                    source_role_documents[role].get(source_id, 0) + written
                )
                entry = _file_identity(output, relative=output_relative)
                role_files[role].append(entry)
                source_map_outputs[role].append({"source_id": source_id, **entry})
                source_entry = source_outputs.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "category": category,
                        "parent_candidate_manifest_sha256": candidate.manifest_sha256,
                        "parent_frozen_validation_manifest_sha256": frozen.manifest_sha256,
                        "audit_attestation_sha256": sha256_file(attestation),
                        "outputs": [],
                    },
                )
                if source_entry["category"] != category:
                    raise DataAuditError(
                        f"source {source_id!r} has conflicting candidate/frozen categories"
                    )
                raw_outputs = source_entry["outputs"]
                assert isinstance(raw_outputs, list)
                raw_outputs.append(entry)
        if not role_files["train"] or not role_files["validation"]:
            raise DataAuditError("filtering removed every train or validation document")

        attribution_output = work / "filtered/attribution/attribution.jsonl"
        attribution_output.parent.mkdir(parents=True, exist_ok=True)
        attribution_rows = 0
        attributed_document_keys: set[tuple[str, str, str]] = set()
        role_tokens = {"train": 0, "validation": 0}
        source_role_tokens: dict[str, dict[str, int]] = {
            "train": {},
            "validation": {},
        }
        with attribution_output.open("w", encoding="utf-8") as output_handle:
            for role, corpus in (("train", candidate), ("validation", frozen)):
                raw_inventory = corpus.value.get("attribution_files")
                if not isinstance(raw_inventory, list):
                    raise DataAuditError("parent attribution inventory is invalid")
                for raw_entry in raw_inventory:
                    if not isinstance(raw_entry, Mapping):
                        raise DataAuditError("parent attribution entry is invalid")
                    relative = _safe_relative(raw_entry.get("path"), "attribution.path")
                    with (corpus.manifest_path.parent / relative).open(
                        "r", encoding="utf-8"
                    ) as handle:
                        for line_number, line in enumerate(handle, start=1):
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as error:
                                raise DataAuditError(
                                    "invalid parent attribution JSONL at "
                                    f"{relative}:{line_number}"
                                ) from error
                            if not isinstance(record, Mapping) or record.get("split") != role:
                                continue
                            key = (
                                role,
                                str(record.get("source_id", "unknown")),
                                str(record.get("text_sha256", "")),
                            )
                            if key not in kept_document_keys:
                                continue
                            if key in attributed_document_keys:
                                raise DataAuditError(
                                    "duplicate parent attribution identity for retained "
                                    f"{role} document {key[2]}"
                                )
                            token_count = record.get("token_count_with_eos")
                            if (
                                isinstance(token_count, bool)
                                or not isinstance(token_count, int)
                                or token_count <= 0
                            ):
                                raise DataAuditError(
                                    "retained attribution token_count_with_eos must be "
                                    "a positive integer"
                                )
                            output_handle.write(
                                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            attributed_document_keys.add(key)
                            attribution_rows += 1
                            role_tokens[role] += token_count
                            source_id = key[1]
                            source_role_tokens[role][source_id] = (
                                source_role_tokens[role].get(source_id, 0) + token_count
                            )
        if has_data_contract:
            missing_attribution = kept_document_keys - attributed_document_keys
            unexpected_attribution = attributed_document_keys - kept_document_keys
            if missing_attribution or unexpected_attribution:
                raise DataAuditError(
                    "filtered data-contract corpus attribution does not cover retained "
                    f"documents exactly (missing={len(missing_attribution)}, "
                    f"unexpected={len(unexpected_attribution)})"
                )
        attribution_files: list[dict[str, object]] = []
        if attribution_rows:
            attribution_files.append(
                _file_identity(
                    attribution_output,
                    relative="filtered/attribution/attribution.jsonl",
                )
            )
        else:
            attribution_output.unlink()

        for inventory in (
            *role_files.values(),
            *source_map_outputs.values(),
            attribution_files,
        ):
            inventory.sort(key=lambda item: str(item["path"]))
        inventories = {
            "train": role_files["train"],
            "validation": role_files["validation"],
            "attribution": attribution_files,
        }
        file_lists: dict[str, dict[str, object]] = {}
        for role, inventory in inventories.items():
            sidecar = work / f"{role}-files.txt"
            atomic_write_text(
                sidecar,
                "".join(f"{item['path']}\n" for item in inventory),
            )
            file_lists[role] = _file_identity(sidecar, relative=sidecar.name)

        sources = []
        for item in sorted(source_outputs.values(), key=lambda entry: str(entry["source_id"])):
            outputs = item.pop("outputs")
            source_id = str(item["source_id"])
            sources.append(
                {
                    **item,
                    "actual_train_tokens": source_role_tokens["train"].get(source_id, 0),
                    "actual_validation_tokens": source_role_tokens["validation"].get(
                        source_id, 0
                    ),
                    "train_rows": source_role_documents["train"].get(source_id, 0),
                    "validation_rows": source_role_documents["validation"].get(source_id, 0),
                    "chunks": [
                        {
                            "shard_id": "audit-filtered",
                            "outputs": outputs,
                            "statistics": {},
                        }
                    ],
                }
            )
        attestation_fingerprint = str(value["attestation_fingerprint"])
        contract_identity: dict[str, object] = {}
        if has_data_contract:
            raw_parent_mix = candidate.value.get("source_mix")
            if not isinstance(raw_parent_mix, Mapping):
                raise DataAuditError("candidate source_mix contract is invalid")
            raw_mix_sources = raw_parent_mix.get("sources")
            if not isinstance(raw_mix_sources, list) or not raw_mix_sources:
                raise DataAuditError("candidate source_mix source inventory is invalid")
            retained_train_source_ids = set(source_role_documents["train"])
            output_source_ids = set(source_outputs)
            mix_source_ids: set[str] = set()
            filtered_mix_sources: list[dict[str, object]] = []
            for index, raw_source in enumerate(raw_mix_sources):
                if not isinstance(raw_source, Mapping):
                    raise DataAuditError(
                        f"candidate source_mix source {index} is invalid"
                    )
                source_id = raw_source.get("source_id")
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or source_id in mix_source_ids
                ):
                    raise DataAuditError(
                        f"candidate source_mix source_id is invalid/duplicate: {source_id!r}"
                    )
                mix_source_ids.add(source_id)
                retained_tokens = source_role_tokens["train"].get(source_id, 0)
                if source_id not in retained_train_source_ids or retained_tokens <= 0:
                    raise DataAuditError(
                        "filtering removed every train document from contracted source "
                        f"{source_id!r}; refusing to renormalize the declared source mix"
                    )
                copied_source = json.loads(
                    json.dumps(
                        raw_source,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                copied_source["actual_train_tokens"] = retained_tokens
                filtered_mix_sources.append(copied_source)
            if retained_train_source_ids != mix_source_ids:
                raise DataAuditError(
                    "filtered train source ownership differs from candidate source_mix"
                )
            if output_source_ids != mix_source_ids:
                raise DataAuditError(
                    "frozen validation introduces source ownership outside the "
                    "candidate source_mix"
                )

            source_map_unsigned = {
                "schema_version": candidate.value["source_map"]["schema_version"],
                "algorithm": candidate.value["source_map"]["algorithm"],
                "roles": source_map_outputs,
            }
            source_map = {
                **source_map_unsigned,
                "fingerprint": _canonical_sha256(source_map_unsigned),
            }
            source_mix_unsigned = {
                "schema_version": raw_parent_mix.get("schema_version"),
                "algorithm": raw_parent_mix.get("algorithm"),
                "unit": raw_parent_mix.get("unit"),
                "basis_points_total": raw_parent_mix.get("basis_points_total"),
                "profile": raw_parent_mix.get("profile"),
                "sources": filtered_mix_sources,
            }
            source_mix = {
                **source_mix_unsigned,
                "fingerprint": _canonical_sha256(source_mix_unsigned),
            }

            raw_format_audit = candidate.value.get("format_audit")
            raw_license_audit = candidate.value.get("license_audit")
            raw_materialization_audit = candidate.value.get("materialization_audit")
            if not all(
                isinstance(item, Mapping)
                for item in (
                    raw_format_audit,
                    raw_license_audit,
                    raw_materialization_audit,
                )
            ):
                raise DataAuditError("candidate data-contract lineage is invalid")
            parent_format_audit = json.loads(
                json.dumps(raw_format_audit, ensure_ascii=False, sort_keys=True)
            )
            parent_license_audit = json.loads(
                json.dumps(raw_license_audit, ensure_ascii=False, sort_keys=True)
            )
            parent_materialization_audit = json.loads(
                json.dumps(raw_materialization_audit, ensure_ascii=False, sort_keys=True)
            )
            frozen_format = frozen.value.get("format_audit")
            frozen_license = frozen.value.get("license_audit")
            frozen_materialization = frozen.value.get("materialization_audit")
            projection_identity = {
                "method": "complete-audit-rejection-ledger-projection-v1",
                "parent_candidate_manifest_sha256": candidate.manifest_sha256,
                "parent_frozen_validation_manifest_sha256": frozen.manifest_sha256,
                "audit_attestation_sha256": sha256_file(attestation),
                "audit_attestation_fingerprint": attestation_fingerprint,
                "rejection_ledger": json.loads(
                    json.dumps(
                        value["rejection_ledger"],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
            }
            format_audit = {
                **parent_format_audit,
                "complete": True,
                "projection": projection_identity,
                "filtered_outputs": {
                    role: [
                        {
                            "source_id": str(item["source_id"]),
                            "path": str(item["path"]),
                            "size": int(item["size"]),
                            "sha256": str(item["sha256"]),
                        }
                        for item in source_map_outputs[role]
                    ]
                    for role in ("train", "validation")
                },
                "frozen_validation_parent_audit": (
                    json.loads(
                        json.dumps(frozen_format, ensure_ascii=False, sort_keys=True)
                    )
                    if isinstance(frozen_format, Mapping)
                    else None
                ),
            }
            license_audit = {
                **parent_license_audit,
                "complete": True,
                "parent_attribution_inventory": parent_license_audit.get(
                    "attribution_inventory"
                ),
                "attribution_inventory": file_lists["attribution"],
                "projection": projection_identity,
                "frozen_validation_parent_audit": (
                    json.loads(
                        json.dumps(frozen_license, ensure_ascii=False, sort_keys=True)
                    )
                    if isinstance(frozen_license, Mapping)
                    else None
                ),
            }
            materialization_audit = {
                "complete": True,
                "network_policy": "offline-audit-materialization",
                **projection_identity,
                "parent_candidate_audit": parent_materialization_audit,
                "parent_frozen_validation_audit": (
                    json.loads(
                        json.dumps(
                            frozen_materialization,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    if isinstance(frozen_materialization, Mapping)
                    else None
                ),
                "sources": [
                    {
                        "source_id": source_id,
                        "method": "jsonl_rejection_ledger_projection",
                        "train_output_count": sum(
                            item["source_id"] == source_id
                            for item in source_map_outputs["train"]
                        ),
                        "validation_output_count": sum(
                            item["source_id"] == source_id
                            for item in source_map_outputs["validation"]
                        ),
                    }
                    for source_id in sorted(mix_source_ids)
                ],
            }
            contract_identity = {
                "source_map": source_map,
                "source_mix": source_mix,
                "format_audit": format_audit,
                "license_audit": license_audit,
                "materialization_audit": materialization_audit,
            }
        identity = {
            "recipe_id": str(candidate.value.get("recipe_id")),
            "recipe_sha256": str(candidate.value.get("recipe_sha256")),
            "resolved_source_lock_sha256": str(candidate.value.get("resolved_source_lock_sha256")),
            "tokenizer_manifest_sha256": str(candidate.value.get("tokenizer_manifest_sha256")),
            "extractor_source_sha256": AUDIT_SOURCE_SHA256,
            "profile": f"{candidate.value.get('profile')}-audit-filtered-{attestation_fingerprint[:12]}",
            "sources": sources,
            "train_files": inventories["train"],
            "validation_files": inventories["validation"],
            "attribution_files": inventories["attribution"],
            "file_lists": file_lists,
            **contract_identity,
        }
        corpus_fingerprint = _canonical_sha256(identity)
        parent_audits = candidate.value.get("audits")
        audits = dict(parent_audits) if isinstance(parent_audits, Mapping) else {}
        audits.update(
            {
                "audit_rejection_ledger_materialization": "complete",
                "train_vs_frozen_validation_exact_dedup": "pending_reaudit_filtered_output",
                "cross_source_exact_dedup": "pending_reaudit_filtered_output",
                "cross_source_near_dedup": "pending_reaudit_filtered_output",
                "full_contextual_pii_scan": "pending_reaudit_filtered_output",
                "project_benchmark_13gram_scan": "pending_reaudit_filtered_output",
            }
        )
        manifest_value = {
            "schema_version": 1,
            "kind": "twen_extracted_base_jsonl_corpus",
            **identity,
            "corpus_fingerprint": corpus_fingerprint,
            "actual_train_tokens": role_tokens["train"] or None,
            "actual_validation_tokens": role_tokens["validation"] or None,
            "actual_train_documents": role_documents["train"],
            "actual_validation_documents": role_documents["validation"],
            "rejected_train_documents": rejected_documents["train"],
            "rejected_validation_documents": rejected_documents["validation"],
            "rejection_reasons": rejection_reasons,
            "network_policy": "offline-audit-materialization",
            "audits": audits,
            "ready_for_data_prepare": True,
            "ready_for_training": False,
        }
        manifest = work / "corpus-manifest.json"
        atomic_write_json(manifest, manifest_value)
        atomic_write_json(
            work / "COMPLETE",
            {
                "schema_version": 1,
                "kind": "twen_extracted_base_jsonl_complete",
                "corpus_fingerprint": corpus_fingerprint,
                "manifest": manifest.name,
                "manifest_sha256": sha256_file(manifest),
                "file_lists": file_lists,
                "ready_for_training": False,
            },
        )
        os.replace(work, root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return root / "corpus-manifest.json"


def audit_lineage_for_role(
    attestation_path: str | Path,
    *,
    extracted_manifest_path: str | Path,
    role: str,
) -> dict[str, object]:
    value = validate_base_audit_attestation(attestation_path)
    identity_name = "candidate" if role == "train" else "frozen_validation"
    identity = value[identity_name]
    assert isinstance(identity, Mapping)
    actual = Path(extracted_manifest_path).resolve()
    if Path(str(identity["manifest_path"])).resolve() != actual:
        raise DataAuditError(f"audit attestation does not bind this extracted {role} manifest")
    gates = value["gates"]
    assert isinstance(gates, Mapping)
    return {
        "path": str(Path(attestation_path).resolve()),
        "sha256": sha256_file(attestation_path),
        "attestation_fingerprint": value["attestation_fingerprint"],
        "bound_as": identity_name,
        "gates": json.loads(json.dumps(gates, sort_keys=True)),
        "ready_for_training": value["ready_for_training"],
    }


__all__ = [
    "AUDIT_KIND",
    "AUDIT_SCHEMA_VERSION",
    "AUDIT_SOURCE_SHA256",
    "BENCHMARK_REGISTRY_KIND",
    "BENCHMARK_REGISTRY_SCHEMA_VERSION",
    "DataAuditError",
    "audit_lineage_for_role",
    "build_base_audit_attestation",
    "inspect_benchmark_registry",
    "materialize_filtered_base_corpus",
    "validate_base_audit_attestation",
]
