"""精确直方图代数：有限桶网格、零桶切分、公共粗化与最小耗尽秩分位。

约定
====

* 每个来源是**一叶直方图**，由严格递增的有限边界 ``edges`` 与各左闭右开桶
  ``[edges[i], edges[i+1])`` 的非负整数计数 ``counts`` 构成，末尾可选一个
  ``[edges[-1], +inf)`` 尾桶（``tail_present`` / ``tail_count``）。
* 没有尾桶的叶表示“最后边界以上没有观测”，等价于一个计数为 0 的开放尾桶；
  内部一律保留这个隐式尾段，``tail_present`` 只决定序列化时是否写出 ``"+inf"``。
* ``null_count`` 单独保存，**不进入**任何分位分母。分位分母只累计非负整数
  计数，因此 ``total() == sum(counts) + tail_count``。
* 边界内部用 ``fractions.Fraction`` 表示，保证 ``10``、``0.1`` 与单位换算后的
  比较都是精确有理数，不受二进制浮点误差影响；输出时整数值还原为 ``int``，
  非整数值写成 ``float``。无穷只出现在尾段，绝不参与数值比较。
* 目标边界落在**非空**源桶内部时无法精确拆桶，直接抛 ``BoundsError``；
  落在**零计数**桶内部可以安全切分，计数不变。本模块从不插值、不摊平。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Sequence, Tuple

from .histogram_errors import (
    BoundsError,
    EmptyPopulationError,
    HistogramError,
    InvalidHistogramError,
    UnsupportedUnitError,
)

#: fixture 中表示正无穷的边界字符串。
INF = "+inf"

#: 单位登记表：1 个该单位等于多少 *us*。换算一律用精确有理数。
UNIT_SCALE = {
    "ns": Fraction(1, 1000),
    "us": Fraction(1),
    "ms": Fraction(1000),
    "s": Fraction(1_000_000),
}


# ---------------------------------------------------------------------------
# 标量与边界
# ---------------------------------------------------------------------------

def decode_edge(value):
    """把 fixture 风格边界解码为 ``Fraction``；``"+inf"`` 原样返回 ``INF``。"""
    if isinstance(value, str):
        if value.strip().lower() in ("+inf", "inf", "+infinity", "infinity"):
            return INF
        raise InvalidHistogramError(f"边界必须是数字或 '+inf'：{value!r}")
    if isinstance(value, bool):
        raise InvalidHistogramError(f"边界必须是数字或 '+inf'：{value!r}")
    if isinstance(value, Fraction):
        return value
    if not isinstance(value, (int, float)):
        raise InvalidHistogramError(f"边界必须是数字或 '+inf'：{value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidHistogramError("只有尾桶上界允许 '+inf'；边界必须是有限值")
    return Fraction(value) if isinstance(value, int) else Fraction(str(value))


def encode_edge(value):
    """把内部边界编码为 JSON 形式；整数值写 int，有理小数值写 float。"""
    if value == INF:
        return INF
    if not isinstance(value, Fraction):
        value = Fraction(value)
    return int(value) if value.denominator == 1 else float(value)


def encode_edges(edges: Iterable) -> list:
    return [encode_edge(edge) for edge in edges]


def decode_target_edges(values: Sequence) -> Tuple[Fraction, ...]:
    """解码并校验一组有限目标边界（严格递增、非空、不含无穷）。"""
    decoded = []
    for raw in values:
        edge = decode_edge(raw)
        if edge == INF:
            raise InvalidHistogramError("目标边界不能是 '+inf'；尾桶上界由来源决定")
        decoded.append(edge)
    if not decoded:
        raise InvalidHistogramError("目标边界不能为空")
    for left, right in zip(decoded, decoded[1:]):
        if not right > left:
            raise InvalidHistogramError("目标边界必须严格递增")
    return tuple(decoded)


def require_count(value, label="计数"):
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidHistogramError(f"{label}必须是非负整数，得到 {value!r}")
    if value < 0:
        raise InvalidHistogramError(f"{label}必须是非负整数，得到 {value!r}")
    return value


def require_unit(unit):
    if not isinstance(unit, str) or not unit:
        raise InvalidHistogramError("unit 必须是非空字符串")
    return unit


def ceil_rank(p, total: int) -> int:
    """按 ``ceil(p*N)`` 1 基秩定义计算秩，使用精确有理数。"""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise HistogramError(f"分位概率必须是 0 到 1 之间的数字：{p!r}")
    if not 0 <= p <= 1:
        raise HistogramError(f"分位概率必须落在 [0, 1]：{p!r}")
    if total <= 0:
        raise EmptyPopulationError("空总体没有分位值")
    exact = Fraction(str(p)) * total
    # Fraction 的 ceil：-(-num // den)
    rank = -(-exact.numerator // exact.denominator)
    return min(max(rank, 1), total)


# ---------------------------------------------------------------------------
# 直方图
# ---------------------------------------------------------------------------

class Histogram:
    """不可变的一叶直方图。

    参数
    ----
    edges : 有限、严格递增的边界序列，至少一个元素。
    counts : 长度 ``len(edges) - 1`` 的非负整数序列。
    tail_count : ``[edges[-1], +inf)`` 的整数计数，默认 0。
    tail_present : 是否在序列化中声明 ``+inf`` 尾桶。
    """

    __slots__ = (
        "edges",
        "counts",
        "tail_count",
        "tail_present",
        "null_count",
        "unit",
    )

    def __init__(
        self,
        edges: Sequence,
        counts: Sequence,
        *,
        tail_count: int = 0,
        tail_present: bool = False,
        null_count: int = 0,
        unit: str = "us",
    ):
        decoded = []
        for raw in edges:
            edge = decode_edge(raw)
            if edge == INF:
                raise InvalidHistogramError("边界序列本身不能包含 '+inf'；请用 tail_present")
            decoded.append(edge)
        if not decoded:
            raise InvalidHistogramError("edges 必须至少包含一个有限边界")
        for left, right in zip(decoded, decoded[1:]):
            if not right > left:
                raise InvalidHistogramError("edges 必须严格递增")
        if not isinstance(counts, (list, tuple)):
            raise InvalidHistogramError("counts 必须是序列")
        if len(counts) != len(decoded) - 1:
            raise InvalidHistogramError(
                f"counts 长度必须等于 len(edges)-1：{len(counts)} != {len(decoded) - 1}"
            )
        tail = require_count(tail_count, "尾桶计数")
        present = bool(tail_present)
        if not present and tail != 0:
            raise InvalidHistogramError("未声明 +inf 尾桶时尾桶计数必须为 0")
        self.edges = tuple(decoded)
        self.counts = tuple(require_count(value) for value in counts)
        self.tail_count = tail
        self.tail_present = present
        self.null_count = require_count(null_count, "null_count")
        self.unit = require_unit(unit)

    # ---- 视图 -------------------------------------------------------
    def segments(self) -> Tuple[Tuple[Fraction, Optional[Fraction], int], ...]:
        """所有桶，含**始终存在**的开放尾段；尾段上界用 ``None`` 表示。"""
        result = [
            (self.edges[i], self.edges[i + 1], self.counts[i])
            for i in range(len(self.counts))
        ]
        result.append((self.edges[-1], None, self.tail_count))
        return tuple(result)

    def visible_segments(self):
        """序列化时会出现的桶；没有声明尾桶时省略隐式零尾段。"""
        result = [
            (self.edges[i], self.edges[i + 1], self.counts[i])
            for i in range(len(self.counts))
        ]
        if self.tail_present:
            result.append((self.edges[-1], None, self.tail_count))
        return tuple(result)

    def finite_total(self) -> int:
        return sum(self.counts)

    def total(self) -> int:
        """分位分母：非 null 的整数计数之和，与 ``null_count`` 无关。"""
        return sum(self.counts) + self.tail_count

    # ---- 切分判定 ---------------------------------------------------
    def splittable(self, point: Fraction) -> bool:
        """``point`` 能否作为精确切分点。

        * 位于支撑下界之前 → 空区域，可切。
        * 与现有边界重合 → 可切。
        * 落在桶内部 → 仅当该桶计数为 0 时可切。
        """
        if point < self.edges[0]:
            return True
        if point in self.edges:
            return True
        if point > self.edges[-1]:
            return self.tail_count == 0
        index = bisect_right(self.edges, point) - 1
        return self.counts[index] == 0

    def lost_below(self, point: Fraction) -> int:
        """完全位于 ``point`` 左侧的非零计数；用于拒绝会丢弃数据的低边界。"""
        lost = 0
        for left, right, count in self.segments():
            if count == 0:
                continue
            if right is not None and right <= point:
                lost += count
        return lost

    # ---- 序列化 -----------------------------------------------------
    def to_dict(self) -> dict:
        edges = list(self.edges)
        counts = list(self.counts)
        if self.tail_present:
            edges.append(INF)
            counts.append(self.tail_count)
        return {
            "unit": self.unit,
            "edges": encode_edges(edges),
            "counts": counts,
            "null_count": self.null_count,
        }

    def __repr__(self) -> str:  # pragma: no cover - 便于调试
        return (
            f"Histogram(edges={encode_edges(self.edges)}, counts={self.counts}, "
            f"tail={self.tail_count}, tail_present={self.tail_present}, unit={self.unit!r})"
        )


def histogram_from_dict(data: dict) -> Histogram:
    """按 fixtures 的写法解析一叶直方图。"""
    if not isinstance(data, dict):
        raise InvalidHistogramError("直方图必须是对象")
    if "edges" not in data or "counts" not in data:
        raise InvalidHistogramError("直方图必须包含 edges 与 counts")
    edges = list(data["edges"])
    counts = list(data["counts"])
    if not isinstance(counts, list):
        raise InvalidHistogramError("counts 必须是数组")
    if not edges:
        raise InvalidHistogramError("edges 不能为空")
    tail_present = decode_edge(edges[-1]) == INF
    if tail_present:
        # fixtures 写法：[0, 20, "+inf"] 配 counts [0, 2]，即 counts 与桶一一对应，
        # 最后一项是 [20, +inf) 的计数，故 len(counts) == len(edges) - 1。
        if len(counts) != len(edges) - 1:
            raise InvalidHistogramError(
                "含 '+inf' 的 edges 要求 len(counts) == len(edges)-1（末项为尾桶计数）"
            )
        return Histogram(
            edges[:-1],
            counts[:-1],
            tail_count=counts[-1],
            tail_present=True,
            null_count=data.get("null_count", 0),
            unit=data.get("unit", "us"),
        )
    return Histogram(
        edges,
        counts,
        tail_count=0,
        tail_present=False,
        null_count=data.get("null_count", 0),
        unit=data.get("unit", "us"),
    )


# ---------------------------------------------------------------------------
# 单位
# ---------------------------------------------------------------------------

def normalize_units(
    histograms: Sequence[Histogram], unit: Optional[str] = None
) -> Tuple[Histogram, ...]:
    """按登记表把各叶的**边界**换到同一单位；计数保持整数不变。

    未指定目标单位时选登记表里最细的单位，使整数边界放大后仍为整数。
    """
    if not histograms:
        return ()
    for histogram in histograms:
        if histogram.unit not in UNIT_SCALE:
            raise UnsupportedUnitError(
                f"单位 {histogram.unit!r} 未登记，无法做可审计的精确换算"
            )
    if unit is None:
        unit = min((histogram.unit for histogram in histograms), key=lambda u: UNIT_SCALE[u])
    if unit not in UNIT_SCALE:
        raise UnsupportedUnitError(f"目标单位 {unit!r} 未登记")
    converted = []
    for histogram in histograms:
        factor = UNIT_SCALE[histogram.unit] / UNIT_SCALE[unit]
        if factor == 1:
            converted.append(histogram)
            continue
        converted.append(
            Histogram(
                [edge * factor for edge in histogram.edges],
                histogram.counts,
                tail_count=histogram.tail_count,
                tail_present=histogram.tail_present,
                null_count=histogram.null_count,
                unit=unit,
            )
        )
    return tuple(converted)


# ---------------------------------------------------------------------------
# 粗化与合并
# ---------------------------------------------------------------------------

def coarsen(histogram: Histogram, target_edges: Sequence) -> Histogram:
    """把一叶精确粗化到目标网格；绝不插值。

    目标边界必须与源边界重合、位于支撑下界之前、或落在零计数桶内部。
    若某个边界切开非空桶，抛 ``BoundsError``。目标网格最后一个桶是
    ``[targets[-1], +inf)``，它只能包含源网格中位于其上的完整桶。
    """
    targets = decode_target_edges(target_edges)
    for point in targets:
        if not histogram.splittable(point):
            raise BoundsError(
                f"目标边界 {encode_edge(point)!r} 切开非空源桶，无法精确拆桶"
            )
    if histogram.lost_below(targets[0]):
        raise BoundsError(
            f"目标下界 {encode_edge(targets[0])!r} 之下存在非零计数，粗化会丢弃数据"
        )

    result_counts = [0] * (len(targets) - 1)
    result_tail = 0
    for left, right, count in histogram.segments():
        if count == 0:
            continue
        if left < targets[0]:
            # 非空且完全位于下界之下：上面已经拒绝；不应到达这里。
            raise BoundsError("内部错误：非空桶位于目标下界之下")
        index = bisect_right(targets, left) - 1
        if index == len(targets) - 1:
            result_tail += count
            continue
        upper = targets[index + 1]
        if right is not None and right > upper:
            raise BoundsError("内部错误：非空桶跨越目标边界")
        result_counts[index] += count

    return Histogram(
        targets,
        result_counts,
        tail_count=result_tail,
        # 源声明了开放尾域，或尾段实际承载了计数，都必须保留 +inf 标签。
        tail_present=histogram.tail_present or result_tail > 0,
        null_count=histogram.null_count,
        unit=histogram.unit,
    )


def common_coarsening(histograms: Sequence[Histogram]) -> Tuple[Fraction, ...]:
    """所有来源共同边界构成的最细公共粗化。

    取各叶有限边界的交集，再补上全局最小下界：该点低于或等于每个叶的支撑
    下界，因此对每个叶都是可切的空区域。交集本身已是每个叶边界的子集。
    """
    if not histograms:
        raise HistogramError("没有来源可供粗化")
    bottom = min(histogram.edges[0] for histogram in histograms)
    shared = set(histograms[0].edges)
    for histogram in histograms[1:]:
        shared &= set(histogram.edges)
    shared.discard(bottom)
    return tuple([bottom] + sorted(shared))


def merge_histograms(histograms: Sequence[Histogram]) -> Histogram:
    """把同一粗化网格、同一单位的各叶按桶相加。

    不同叶的尾桶声明可以不同；合并后的域只要有任一叶开放就保持开放。
    """
    if not histograms:
        raise HistogramError("没有可合并的叶")
    reference = histograms[0]
    for other in histograms[1:]:
        if other.unit != reference.unit:
            raise UnsupportedUnitError("不同单位的叶不能直接按桶相加")
        if other.edges != reference.edges:
            raise HistogramError("只有落在同一粗化网格上的叶才能直接相加")
    counts = [
        sum(other.counts[index] for other in histograms)
        for index in range(len(reference.counts))
    ]
    return Histogram(
        reference.edges,
        counts,
        tail_count=sum(other.tail_count for other in histograms),
        tail_present=any(other.tail_present for other in histograms),
        null_count=sum(other.null_count for other in histograms),
        unit=reference.unit,
    )


def auto_rollup(
    histograms: Sequence[Histogram], unit: Optional[str] = None
) -> Histogram:
    """自动精确汇总：先统一单位，再按公共粗化网格逐桶相加。"""
    normalized = normalize_units(histograms, unit)
    grid = common_coarsening(normalized)
    return merge_histograms([coarsen(histogram, grid) for histogram in normalized])


def manual_rollup(
    histograms: Sequence[Histogram],
    target_edges: Sequence,
    unit: Optional[str] = None,
) -> Histogram:
    """严格模式的手动边界汇总；任何不可精确切分都抛 ``BoundsError``。"""
    normalized = normalize_units(histograms, unit)
    return merge_histograms([coarsen(histogram, target_edges) for histogram in normalized])


# ---------------------------------------------------------------------------
# 分位
# ---------------------------------------------------------------------------

def cumulative_bucket(histogram: Histogram, rank: int):
    """最小累计达到 ``rank`` 的桶；零计数桶不会被命中。"""
    if not 1 <= rank <= histogram.total():
        raise EmptyPopulationError("秩超出非空总体")
    accumulated = 0
    for left, right, count in histogram.segments():
        accumulated += count
        if accumulated >= rank:
            return left, right, count
    raise AssertionError("累计计数与 total 不一致")


@dataclass(frozen=True)
class QuantileRange:
    """一个分位的**桶半开值域**，不是点估计。"""

    probability: float
    rank: int
    population: int
    lower: Fraction
    upper: Optional[Fraction]  # None 表示 +inf
    bucket_count: int

    @property
    def upper_is_infinite(self) -> bool:
        return self.upper is None

    def to_dict(self) -> dict:
        return {
            "probability": self.probability,
            "rank": self.rank,
            "population": self.population,
            "lower": encode_edge(self.lower),
            "upper": INF if self.upper is None else encode_edge(self.upper),
            "upper_is_infinite": self.upper_is_infinite,
            "bucket_count": self.bucket_count,
            "definition": "smallest bucket whose cumulative count first reaches ceil(p*N)",
            "note": "bucket bound, not a point estimate",
        }


def quantile_range(histogram: Histogram, probability) -> QuantileRange:
    total = histogram.total()
    rank = ceil_rank(probability, total)
    left, right, count = cumulative_bucket(histogram, rank)
    return QuantileRange(
        probability=float(probability),
        rank=rank,
        population=total,
        lower=left,
        upper=right,
        bucket_count=count,
    )


def quantile_report(histogram: Histogram, probabilities: Sequence) -> list:
    """空总体返回 ``defined: false``，而不是 NaN 或伪造值域。"""
    reports = []
    for probability in probabilities:
        if histogram.total() <= 0:
            reports.append(
                {
                    "probability": float(probability),
                    "defined": False,
                    "reason": "empty_population",
                    "population": 0,
                    "null_count": histogram.null_count,
                    "lower": None,
                    "upper": None,
                    "upper_is_infinite": False,
                }
            )
            continue
        item = quantile_range(histogram, probability).to_dict()
        item["defined"] = True
        item["null_count"] = histogram.null_count
        reports.append(item)
    return reports
