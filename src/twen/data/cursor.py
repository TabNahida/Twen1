"""World-size-independent deterministic global data cursor."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace

CURSOR_SCHEMA_VERSION = 2
SHUFFLE_ALGORITHM = "shard-local-affine-v1"
QUALITY_COOLDOWN_CURSOR_SCHEMA_VERSION = 1
QUALITY_COOLDOWN_CURSOR_KIND = "deterministic-two-phase-quality-cooldown"


def _stable_integer(*parts: object) -> int:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Flat deterministic index over immutable source shards."""

    shard_ids: tuple[str, ...]
    shard_lengths: tuple[int, ...]
    fingerprint: str
    _cumulative_ends: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.shard_ids or len(self.shard_ids) != len(self.shard_lengths):
            raise ValueError("shard_ids and shard_lengths must be non-empty and aligned")
        if len(set(self.shard_ids)) != len(self.shard_ids):
            raise ValueError("shard IDs must be unique")
        if any(length <= 0 for length in self.shard_lengths):
            raise ValueError("all shard lengths must be positive")
        if not self.fingerprint:
            raise ValueError("dataset fingerprint is required")
        total = 0
        ends: list[int] = []
        for length in self.shard_lengths:
            total += length
            ends.append(total)
        object.__setattr__(self, "_size", total)
        object.__setattr__(self, "_cumulative_ends", tuple(ends))

    @classmethod
    def from_shards(
        cls,
        shards: Mapping[str, int] | Iterable[tuple[str, int]],
        *,
        fingerprint: str | None = None,
    ) -> DatasetLayout:
        items = list(shards.items() if isinstance(shards, Mapping) else shards)
        ids = tuple(item[0] for item in items)
        lengths = tuple(int(item[1]) for item in items)
        if fingerprint is None:
            canonical = json.dumps(items, separators=(",", ":"), ensure_ascii=False)
            fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(ids, lengths, fingerprint)

    @property
    def size(self) -> int:
        return self._size

    @property
    def cumulative_ends(self) -> tuple[int, ...]:
        return self._cumulative_ends

    def locate(self, flat_index: int) -> tuple[str, int]:
        if not 0 <= flat_index < self.size:
            raise IndexError(flat_index)
        ends = self.cumulative_ends
        shard_index = bisect.bisect_right(ends, flat_index)
        start = 0 if shard_index == 0 else ends[shard_index - 1]
        return self.shard_ids[shard_index], flat_index - start


def _affine_permutation(index: int, size: int, seed: int, epoch: int) -> int:
    """Stateless O(1) epoch permutation with no billion-element shuffle table."""

    if size == 1:
        return 0
    multiplier = _stable_integer("multiplier", seed, epoch) % size
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = _stable_integer("offset", seed, epoch) % size
    return (multiplier * index + offset) % size


@dataclass(frozen=True, slots=True)
class SampleReference:
    global_position: int
    epoch: int
    epoch_position: int
    flat_index: int
    shard_id: str
    shard_offset: int


@dataclass(frozen=True, slots=True)
class GlobalCursorState:
    schema_version: int
    dataset_fingerprint: str
    dataset_size: int
    seed: int
    next_global_sample: int
    committed_tokens: int
    shuffle: bool
    shuffle_algorithm: str = SHUFFLE_ALGORITHM

    def __post_init__(self) -> None:
        if self.schema_version != CURSOR_SCHEMA_VERSION:
            raise ValueError("unsupported cursor schema")
        if not self.dataset_fingerprint or self.dataset_size <= 0:
            raise ValueError("cursor dataset identity/size are invalid")
        if self.next_global_sample < 0 or self.committed_tokens < 0:
            raise ValueError("cursor counters cannot be negative")
        if not isinstance(self.shuffle, bool):
            raise TypeError("cursor shuffle must be a bool")
        if self.shuffle_algorithm != SHUFFLE_ALGORITHM:
            raise ValueError(f"unsupported cursor shuffle algorithm {self.shuffle_algorithm!r}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GlobalCursorState:
        shuffle = payload["shuffle"]
        if not isinstance(shuffle, bool):
            raise TypeError("cursor shuffle must be a bool")
        return cls(
            schema_version=int(payload["schema_version"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            dataset_size=int(payload["dataset_size"]),
            seed=int(payload["seed"]),
            next_global_sample=int(payload["next_global_sample"]),
            committed_tokens=int(payload["committed_tokens"]),
            shuffle=shuffle,
            shuffle_algorithm=str(payload.get("shuffle_algorithm", "legacy-global-affine")),
        )


class DeterministicGlobalCursor:
    """Plans global batches; only ``commit`` advances durable state.

    Every rank reconstructs the same global batch and selects its own slice.
    Consequently a checkpoint can resume with another world size without
    changing the global sample order.
    """

    def __init__(
        self,
        layout: DatasetLayout,
        *,
        seed: int,
        next_global_sample: int = 0,
        committed_tokens: int = 0,
        shuffle: bool = True,
    ) -> None:
        if next_global_sample < 0 or committed_tokens < 0:
            raise ValueError("cursor counters cannot be negative")
        self.layout = layout
        self.seed = seed
        self.next_global_sample = next_global_sample
        self.committed_tokens = committed_tokens
        self.shuffle = shuffle
        self._epoch_plans: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def _epoch_shard_plan(self, epoch: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return a shuffled shard order and its cumulative logical lengths.

        Shuffling one flat billion-sample index makes nearly every consecutive
        sample land in another mmap file.  A full permutation does not require
        that I/O-hostile order: shuffle shard blocks first, then independently
        permute records inside each block.  This preserves exact resume/world-
        size semantics while keeping each shard handle hot for its whole block.
        """

        cached = self._epoch_plans.get(epoch)
        if cached is not None:
            return cached
        count = len(self.layout.shard_ids)
        order = tuple(
            _affine_permutation(position, count, self.seed, epoch) for position in range(count)
        )
        total = 0
        ends: list[int] = []
        for shard_index in order:
            total += self.layout.shard_lengths[shard_index]
            ends.append(total)
        result = (order, tuple(ends))
        # Training budgets normally span only a handful of corpus epochs. Keep
        # the tiny O(num_shards) plans so sample_at remains O(log num_shards).
        self._epoch_plans[epoch] = result
        return result

    def sample_at(self, global_position: int) -> SampleReference:
        if global_position < 0:
            raise IndexError(global_position)
        epoch, epoch_position = divmod(global_position, self.layout.size)
        if self.shuffle:
            order, logical_ends = self._epoch_shard_plan(epoch)
            logical_shard = bisect.bisect_right(logical_ends, epoch_position)
            logical_start = 0 if logical_shard == 0 else logical_ends[logical_shard - 1]
            shard_index = order[logical_shard]
            shard_length = self.layout.shard_lengths[shard_index]
            local_position = epoch_position - logical_start
            within_seed = _stable_integer(
                "within-shard",
                self.seed,
                epoch,
                self.layout.shard_ids[shard_index],
            )
            shard_offset = _affine_permutation(
                local_position,
                shard_length,
                within_seed,
                epoch,
            )
            original_start = 0 if shard_index == 0 else self.layout.cumulative_ends[shard_index - 1]
            flat_index = original_start + shard_offset
            shard_id = self.layout.shard_ids[shard_index]
        else:
            flat_index = epoch_position
            shard_id, shard_offset = self.layout.locate(flat_index)
        return SampleReference(
            global_position=global_position,
            epoch=epoch,
            epoch_position=epoch_position,
            flat_index=flat_index,
            shard_id=shard_id,
            shard_offset=shard_offset,
        )

    def plan_global_batch(self, global_batch_samples: int) -> tuple[SampleReference, ...]:
        if global_batch_samples <= 0:
            raise ValueError("global_batch_samples must be positive")
        return tuple(
            self.sample_at(position)
            for position in range(
                self.next_global_sample,
                self.next_global_sample + global_batch_samples,
            )
        )

    def plan_rank_batch(
        self,
        global_batch_samples: int,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[SampleReference, ...]:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size are invalid")
        if global_batch_samples % world_size:
            raise ValueError("global_batch_samples must be divisible by world_size")
        batch = self.plan_global_batch(global_batch_samples)
        return tuple(batch[index] for index in range(rank, len(batch), world_size))

    def commit(self, *, global_batch_samples: int, token_count: int) -> None:
        if global_batch_samples <= 0 or token_count <= 0:
            raise ValueError("invalid commit counters")
        self.next_global_sample += global_batch_samples
        self.committed_tokens += token_count

    def state_dict(self) -> dict[str, object]:
        return GlobalCursorState(
            schema_version=CURSOR_SCHEMA_VERSION,
            dataset_fingerprint=self.layout.fingerprint,
            dataset_size=self.layout.size,
            seed=self.seed,
            next_global_sample=self.next_global_sample,
            committed_tokens=self.committed_tokens,
            shuffle=self.shuffle,
            shuffle_algorithm=SHUFFLE_ALGORITHM,
        ).to_dict()

    @classmethod
    def from_state_dict(
        cls, layout: DatasetLayout, payload: Mapping[str, object]
    ) -> DeterministicGlobalCursor:
        state = GlobalCursorState.from_dict(payload)
        if state.schema_version != CURSOR_SCHEMA_VERSION:
            raise ValueError("unsupported cursor schema")
        if state.dataset_fingerprint != layout.fingerprint or state.dataset_size != layout.size:
            raise ValueError("cursor dataset fingerprint/size mismatch")
        return cls(
            layout,
            seed=state.seed,
            next_global_sample=state.next_global_sample,
            committed_tokens=state.committed_tokens,
            shuffle=state.shuffle,
        )


class DeterministicCooldownCursor:
    """Two immutable layouts with one token-coordinate phase transition.

    The primary and cooldown positions are persisted independently. The active
    phase is derived only from globally committed tokens, so changing world
    size changes rank partitioning but never sample order or transition state.
    A whole optimizer batch that starts below the threshold remains primary;
    the next batch switches after the commit crosses the threshold.
    """

    def __init__(
        self,
        primary_layout: DatasetLayout,
        cooldown_layout: DatasetLayout,
        *,
        seed: int,
        cooldown_start_tokens: int,
        shuffle: bool = True,
        _primary_cursor: DeterministicGlobalCursor | None = None,
        _cooldown_cursor: DeterministicGlobalCursor | None = None,
        _next_global_sample: int = 0,
        _committed_tokens: int = 0,
    ) -> None:
        if cooldown_start_tokens <= 0:
            raise ValueError("quality cooldown start tokens must be positive")
        if primary_layout.fingerprint == cooldown_layout.fingerprint:
            raise ValueError("quality cooldown requires distinct dataset fingerprints")
        self.primary_layout = primary_layout
        self.cooldown_layout = cooldown_layout
        self.seed = seed
        self.cooldown_start_tokens = cooldown_start_tokens
        self.shuffle = shuffle
        self._primary_cursor = _primary_cursor or DeterministicGlobalCursor(
            primary_layout, seed=seed, shuffle=shuffle
        )
        self._cooldown_cursor = _cooldown_cursor or DeterministicGlobalCursor(
            cooldown_layout, seed=seed, shuffle=shuffle
        )
        self.next_global_sample = _next_global_sample
        self.committed_tokens = _committed_tokens
        self._validate_state()

    @property
    def active_phase(self) -> str:
        return "cooldown" if self.committed_tokens >= self.cooldown_start_tokens else "primary"

    @property
    def active_layout(self) -> DatasetLayout:
        return self.cooldown_layout if self.active_phase == "cooldown" else self.primary_layout

    @property
    def phase_next_global_sample(self) -> int:
        return self._active_cursor.next_global_sample

    @property
    def _active_cursor(self) -> DeterministicGlobalCursor:
        return self._cooldown_cursor if self.active_phase == "cooldown" else self._primary_cursor

    @property
    def dataset_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": QUALITY_COOLDOWN_CURSOR_KIND,
                "primary": self.primary_layout.fingerprint,
                "cooldown": self.cooldown_layout.fingerprint,
                "start_tokens": self.cooldown_start_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _validate_state(self) -> None:
        cursors = (self._primary_cursor, self._cooldown_cursor)
        if any(cursor.seed != self.seed or cursor.shuffle != self.shuffle for cursor in cursors):
            raise ValueError("quality cooldown phase cursor seed/shuffle mismatch")
        if self._primary_cursor.layout != self.primary_layout:
            raise ValueError("quality cooldown primary layout mismatch")
        if self._cooldown_cursor.layout != self.cooldown_layout:
            raise ValueError("quality cooldown layout mismatch")
        if self.next_global_sample != sum(cursor.next_global_sample for cursor in cursors):
            raise ValueError("quality cooldown global sample counter mismatch")
        if self.committed_tokens != sum(cursor.committed_tokens for cursor in cursors):
            raise ValueError("quality cooldown committed token counter mismatch")
        if self.committed_tokens < self.cooldown_start_tokens and (
            self._cooldown_cursor.next_global_sample != 0
            or self._cooldown_cursor.committed_tokens != 0
        ):
            raise ValueError("quality cooldown cursor advanced before its phase")
        if self._cooldown_cursor.next_global_sample > 0 and (
            self._primary_cursor.committed_tokens < self.cooldown_start_tokens
        ):
            raise ValueError("quality cooldown cursor advanced before primary threshold")

    def plan_global_batch(self, global_batch_samples: int) -> tuple[SampleReference, ...]:
        local = self._active_cursor.plan_global_batch(global_batch_samples)
        return tuple(
            replace(reference, global_position=self.next_global_sample + index)
            for index, reference in enumerate(local)
        )

    def plan_rank_batch(
        self,
        global_batch_samples: int,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[SampleReference, ...]:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size are invalid")
        if global_batch_samples % world_size:
            raise ValueError("global_batch_samples must be divisible by world_size")
        batch = self.plan_global_batch(global_batch_samples)
        return tuple(batch[index] for index in range(rank, len(batch), world_size))

    def commit(self, *, global_batch_samples: int, token_count: int) -> None:
        active = self._active_cursor
        active.commit(
            global_batch_samples=global_batch_samples,
            token_count=token_count,
        )
        self.next_global_sample += global_batch_samples
        self.committed_tokens += token_count
        self._validate_state()

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": QUALITY_COOLDOWN_CURSOR_SCHEMA_VERSION,
            "kind": QUALITY_COOLDOWN_CURSOR_KIND,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_size": self.primary_layout.size + self.cooldown_layout.size,
            "seed": self.seed,
            "next_global_sample": self.next_global_sample,
            "committed_tokens": self.committed_tokens,
            "shuffle": self.shuffle,
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
            "cooldown_start_tokens": self.cooldown_start_tokens,
            "active_phase": self.active_phase,
            "primary_cursor": self._primary_cursor.state_dict(),
            "cooldown_cursor": self._cooldown_cursor.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        primary_layout: DatasetLayout,
        cooldown_layout: DatasetLayout,
        payload: Mapping[str, object],
        *,
        cooldown_start_tokens: int,
    ) -> DeterministicCooldownCursor:
        if (
            int(payload.get("schema_version", -1)) != QUALITY_COOLDOWN_CURSOR_SCHEMA_VERSION
            or payload.get("kind") != QUALITY_COOLDOWN_CURSOR_KIND
        ):
            raise ValueError("unsupported quality cooldown cursor schema")
        if int(payload.get("cooldown_start_tokens", -1)) != cooldown_start_tokens:
            raise ValueError("quality cooldown transition token changed on resume")
        primary_payload = payload.get("primary_cursor")
        cooldown_payload = payload.get("cooldown_cursor")
        if not isinstance(primary_payload, Mapping) or not isinstance(cooldown_payload, Mapping):
            raise ValueError("quality cooldown checkpoint is missing phase cursors")
        primary = DeterministicGlobalCursor.from_state_dict(primary_layout, primary_payload)
        cooldown = DeterministicGlobalCursor.from_state_dict(cooldown_layout, cooldown_payload)
        shuffle = payload.get("shuffle")
        if not isinstance(shuffle, bool):
            raise TypeError("quality cooldown cursor shuffle must be a bool")
        result = cls(
            primary_layout,
            cooldown_layout,
            seed=int(payload["seed"]),
            cooldown_start_tokens=cooldown_start_tokens,
            shuffle=shuffle,
            _primary_cursor=primary,
            _cooldown_cursor=cooldown,
            _next_global_sample=int(payload["next_global_sample"]),
            _committed_tokens=int(payload["committed_tokens"]),
        )
        if payload.get("dataset_fingerprint") != result.dataset_fingerprint:
            raise ValueError("quality cooldown composite dataset fingerprint changed")
        if int(payload.get("dataset_size", -1)) != (primary_layout.size + cooldown_layout.size):
            raise ValueError("quality cooldown composite dataset size changed")
        if payload.get("active_phase") != result.active_phase:
            raise ValueError("quality cooldown active phase disagrees with committed tokens")
        return result


def gradient_accumulation_steps(
    *,
    global_batch_tokens: int,
    world_size: int,
    microbatch_tokens_per_rank: int,
) -> int:
    """Compute exact accumulation needed to preserve global batch tokens."""

    if global_batch_tokens <= 0 or world_size <= 0 or microbatch_tokens_per_rank <= 0:
        raise ValueError("batch geometry values must be positive")
    denominator = world_size * microbatch_tokens_per_rank
    steps, remainder = divmod(global_batch_tokens, denominator)
    if remainder or steps == 0:
        raise ValueError(
            "global_batch_tokens must be an exact positive multiple of "
            "world_size * microbatch_tokens_per_rank"
        )
    return steps
