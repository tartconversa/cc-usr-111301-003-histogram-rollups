"""手动细边界请求的联合可行计数约束。

目标边界落在**非空**源桶内部时无法精确拆桶。汇总层不插值，也不把逐桶
独立上下界相加。

可行域的正确形状
================

设 ``a_j`` 为目标桶 ``j`` 的计数。每个**非空源段** ``s``（计数 ``c_s``）只能
把它的 ``c_s`` 个观测分配到它所覆盖的目标桶里，且必须分配完。于是

    a_j = Σ_s x_{s,j},   x_{s,j} ≥ 0 且 Σ_j x_{s,j} = c_s

由此推出两条必须分开陈述的结论：

1. **总量是精确守恒的**：``Σ_j a_j = Σ_s c_s = population``。任何分配都不能
   改变总量。因此 ``joint_min_total == joint_max_total == population``。
2. **逐桶区间只是单独可达**：桶 ``j`` 的取值可以落在
   ``[fixed_j, fixed_j + Σ_{s 触及 j} c_s]`` 中的任意整数（把份额分给别处即可取
   下界，把触及的份额都留给 ``j`` 即可取上界），但这些下界/上界**不能同时**
   达到。把逐桶下界相加会低于 population，把逐桶上界相加会高于 population——
   这正是“独立上下界相加制造总量漂移”。

模块因此同时输出逐桶区间、精确总量，以及两个独立求和值作为漂移证据。
"""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .histogram_algebra import (
    Histogram,
    decode_target_edges,
    encode_edge,
    normalize_units,
)
from .histogram_errors import BoundsError, HistogramError


@dataclass(frozen=True)
class BucketFeasibility:
    """单个目标桶的可达计数范围；区间内每个整数都单独可达。"""

    lower: int
    upper: int
    split_sources: int

    def to_dict(self) -> dict:
        return {
            "min": self.lower,
            "max": self.upper,
            "split_sources": self.split_sources,
            "attainable_individually": True,
        }


@dataclass(frozen=True)
class FeasibleRollup:
    target_edges: Tuple[Fraction, ...]
    buckets: Tuple[BucketFeasibility, ...]
    population: int
    null_count: int
    witness: Tuple[int, ...]
    unsplittable_edges: Tuple[Fraction, ...]
    sources: Tuple[dict, ...]
    unit: str

    @property
    def exact(self) -> bool:
        """所有目标边界都可精确切分，逐桶计数唯一。"""
        return not self.unsplittable_edges

    @property
    def joint_min_total(self) -> int:
        """联合可达总量下界：对每个可行分配都成立。"""
        return self.population

    @property
    def joint_max_total(self) -> int:
        """联合可达总量上界：等于下界，因为总量守恒。"""
        return self.population

    @property
    def independent_lower_sum(self) -> int:
        """逐桶下界之和；通常小于 population，是漂移的来源之一。"""
        return sum(bucket.lower for bucket in self.buckets)

    @property
    def independent_upper_sum(self) -> int:
        """逐桶上界之和；通常大于 population，是漂移的另一来源。"""
        return sum(bucket.upper for bucket in self.buckets)

    def to_dict(self) -> dict:
        last = len(self.buckets) - 1
        return {
            "kind": "exact" if self.exact else "feasible",
            "exact": self.exact,
            "target_edges": [encode_edge(edge) for edge in self.target_edges],
            "buckets": [
                {
                    "index": index,
                    "lower_edge": encode_edge(self.target_edges[index]),
                    "upper_edge": "+inf" if index == last else encode_edge(self.target_edges[index + 1]),
                    "count_range": bucket.to_dict(),
                }
                for index, bucket in enumerate(self.buckets)
            ],
            "joint_min_total": self.joint_min_total,
            "joint_max_total": self.joint_max_total,
            "population": self.population,
            "null_count": self.null_count,
            "witness_counts": list(self.witness),
            "witness_total": sum(self.witness),
            "independent_lower_sum": self.independent_lower_sum,
            "independent_upper_sum": self.independent_upper_sum,
            "unsplittable_edges": [encode_edge(edge) for edge in self.unsplittable_edges],
            "sources": list(self.sources),
            "unit": self.unit,
            "note": (
                "joint total is population; per-bucket bounds are individually attainable "
                "and must not be summed"
            ),
        }


def _bucket_index(targets: Sequence[Fraction], point: Fraction) -> int:
    """``point`` 所在目标桶的下标；越过最后一个边界归入尾桶。"""
    return max(bisect_right(targets, point) - 1, 0)


def leaf_structure(histogram: Histogram, targets: Sequence[Fraction]):
    """把一叶源段摊到目标桶，返回 ``(fixed, groups, group_counts)``。

    * ``fixed[j]``：完整落在目标桶 ``j`` 内的非空源段计数；
    * ``groups[j]``：触及桶 ``j`` 的跨桶源段编号；
    * ``group_counts[g]``：该跨桶源段的总计数。
    """
    count = len(targets)
    fixed = [0] * count
    groups: List[List[int]] = [[] for _ in range(count)]
    group_counts: Dict[int, int] = {}
    next_group = 0

    for left, right, value in histogram.segments():
        if value == 0:
            continue
        start = _bucket_index(targets, left)
        if right is None:
            end = count - 1
        elif right in targets:
            end = max(_bucket_index(targets, right) - 1, start)
        else:
            end = max(_bucket_index(targets, right), start)
        if start == end:
            fixed[start] += value
            continue
        group = next_group
        next_group += 1
        group_counts[group] = value
        for index in range(start, end + 1):
            groups[index].append(group)
    return fixed, groups, group_counts


def _witness(fixed, groups, group_counts, targets_len):
    """构造一个满足总量守恒的具体分配：份额全部留在其最左覆盖桶。"""
    result = list(fixed)
    placed = set()
    for index in range(targets_len):
        for group in groups[index]:
            if group not in placed:
                result[index] += group_counts[group]
                placed.add(group)
    return result


def feasible_rollup(
    histograms: Sequence[Histogram],
    target_edges: Sequence,
    unit: Optional[str] = None,
) -> FeasibleRollup:
    """跨来源联合可行域；来源彼此独立，逐桶区间按来源相加。"""
    normalized = normalize_units(histograms, unit)
    if not normalized:
        raise HistogramError("没有来源")
    targets = decode_target_edges(target_edges)
    for histogram in normalized:
        if histogram.lost_below(targets[0]):
            raise BoundsError("目标下界之下存在非空计数；可行域请求会丢弃数据")
        # 下界落在非空桶内部时，该桶在边界以下的部分会被丢弃；这不是“无法精确
        # 拆桶”而是信息丢失，必须拒绝。
        if targets[0] > histogram.edges[0] and not histogram.splittable(targets[0]):
            raise BoundsError("目标下界切开非空源桶，边界以下的数据会被丢弃")

    lower = [0] * len(targets)
    upper = [0] * len(targets)
    witness = [0] * len(targets)
    split_flags = [0] * len(targets)
    population = 0
    unsplittable = set()

    for histogram in normalized:
        for point in targets:
            if point >= histogram.edges[0] and not histogram.splittable(point):
                unsplittable.add(point)
        fixed, groups, group_counts = leaf_structure(histogram, targets)
        leaf_witness = _witness(fixed, groups, group_counts, len(targets))
        for index in range(len(targets)):
            lower[index] += fixed[index]
            upper[index] += fixed[index] + sum(group_counts[g] for g in groups[index])
            witness[index] += leaf_witness[index]
            if groups[index]:
                split_flags[index] += 1
        population += histogram.total()

    # 逐桶见证必须守恒且落在逐桶区间内；违反说明建模有误，属于内部错误。
    if sum(witness) != population:
        raise HistogramError(f"内部错误：见证分配总量 {sum(witness)} != population {population}")
    if any(witness[i] > upper[i] or witness[i] < lower[i] for i in range(len(targets))):
        raise HistogramError("内部错误：见证分配不落在逐桶区间内")

    return FeasibleRollup(
        target_edges=targets,
        buckets=tuple(
            BucketFeasibility(lower[i], upper[i], split_flags[i]) for i in range(len(targets))
        ),
        population=population,
        null_count=sum(h.null_count for h in normalized),
        witness=tuple(witness),
        unsplittable_edges=tuple(sorted(unsplittable)),
        sources=tuple(
            {
                "unit": h.unit,
                "population": h.total(),
                "null_count": h.null_count,
                "edges": [encode_edge(e) for e in h.edges],
                "tail_present": h.tail_present,
            }
            for h in normalized
        ),
        unit=normalized[0].unit,
    )
