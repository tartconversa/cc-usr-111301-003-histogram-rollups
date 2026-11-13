"""汇总层可能拒绝输入或请求的错误类型。

这些错误只描述计数与桶边界层面的问题；本包不读取单次请求正文、模型产物、
轨迹或图片，因此错误消息里也不会出现这些内容。
"""


class HistogramError(ValueError):
    """所有汇总层错误的基类。"""


class InvalidHistogramError(HistogramError):
    """来源直方图本身不合法（边界、计数、单位或恒等式不成立）。"""


class BoundsError(HistogramError):
    """目标边界请求无法用给定来源精确表达。"""


class SourceConflictError(HistogramError):
    """同一来源被互相矛盾地登记或重复纳入。"""


class UnknownSourceError(HistogramError):
    """引用了未登记的来源版本。"""


class UnsupportedUnitError(HistogramError):
    """缺少可审计的单位换算，不能把不同单位混入同一坐标系。"""


class EmptyPopulationError(HistogramError):
    """非空分位请求命中了零总数（例如只有 null 的叶）。"""
