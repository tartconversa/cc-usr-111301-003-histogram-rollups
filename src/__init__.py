"""异构延迟直方图的可证精确汇总。

只处理 fixtures 中的合成设备汇总计数；不读取单次请求正文、模型响应、
轨迹或图片。
"""

from .histogram_algebra import (
    Histogram,
    QuantileRange,
    auto_rollup,
    coarsen,
    common_coarsening,
    encode_edge,
    encode_edges,
    histogram_from_dict,
    manual_rollup,
    merge_histograms,
    normalize_units,
    quantile_range,
    quantile_report,
)
from .histogram_feasibility import BucketFeasibility, FeasibleRollup, feasible_rollup
from .histogram_registry import (
    LeafRegistry,
    LeafVersion,
    RollupLedger,
    RollupNode,
    histogram_digest,
)
from .histogram_errors import (
    BoundsError,
    EmptyPopulationError,
    HistogramError,
    InvalidHistogramError,
    SourceConflictError,
    UnknownSourceError,
    UnsupportedUnitError,
)

__all__ = [
    "Histogram",
    "QuantileRange",
    "BucketFeasibility",
    "FeasibleRollup",
    "LeafRegistry",
    "LeafVersion",
    "RollupLedger",
    "RollupNode",
    "HistogramError",
    "InvalidHistogramError",
    "BoundsError",
    "SourceConflictError",
    "UnknownSourceError",
    "UnsupportedUnitError",
    "EmptyPopulationError",
    "auto_rollup",
    "manual_rollup",
    "coarsen",
    "common_coarsening",
    "merge_histograms",
    "normalize_units",
    "quantile_range",
    "quantile_report",
    "feasible_rollup",
    "histogram_from_dict",
    "histogram_digest",
    "encode_edge",
    "encode_edges",
]
