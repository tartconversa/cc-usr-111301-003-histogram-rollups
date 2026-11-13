"""汇总层命令行入口。

    python -m src.histogram_rollup fixtures/histograms.json fixtures/rollups.json
    python -m src.histogram_rollup fixtures/histograms.json fixtures/rollups.json --json

只读入 fixtures 的合成汇总计数；不涉及单次请求正文、模型响应、轨迹或图片。
"""

import argparse
import json
from pathlib import Path

from .histogram_algebra import (
    auto_rollup,
    encode_edge,
    histogram_from_dict,
    quantile_report,
)
from .histogram_feasibility import feasible_rollup
from .histogram_registry import LeafRegistry, RollupLedger


def load_fixtures(histogram_path, rollup_path):
    histograms = json.loads(Path(histogram_path).read_text(encoding="utf-8"))
    rollups = json.loads(Path(rollup_path).read_text(encoding="utf-8"))
    if not isinstance(histograms, dict) or "histograms" not in histograms:
        raise ValueError("直方图文件必须包含 histograms 数组")
    return histograms["histograms"], rollups


def build_report(histogram_items, config):
    registry = LeafRegistry()
    ledger = RollupLedger(registry)
    for item in histogram_items:
        registration = registry.register(
            item["histogram_id"],
            item.get("revision", 0),
            histogram_from_dict(item),
        )
        if registration["status"] == "stale":
            raise ValueError(f"{item['histogram_id']} 的 revision 不是最新")

    for name, entries in config["parents"].items():
        ledger.define_parent(name, entries)

    report = {"groups": {}, "parents": {}}
    for name in config["parents"]:
        node = ledger.build(name)
        group = {
            "leaf_versions": [
                {"histogram_id": leaf["histogram_id"], "revision": leaf["revision"], "digest": leaf["digest"]}
                for leaf in node.leaves
            ],
            "leaf_set_digest": node.leaf_set_digest,
            "node_id": node.node_id,
            "auto": {
                "grid": [encode_edge(edge) for edge in node.histogram.edges],
                "counts": list(node.histogram.counts),
                "tail_count": node.histogram.tail_count,
                "tail_present": node.histogram.tail_present,
                "population": node.histogram.total(),
                "null_count": node.histogram.null_count,
                "conserved": node.histogram.total()
                == sum(ledger.registry.latest(leaf["histogram_id"]).histogram.total() for leaf in node.leaves),
            },
        }
        probabilities = config.get("quantiles", [])
        if probabilities:
            group["quantiles"] = quantile_report(node.histogram, probabilities)
        report["groups"][name] = group
        report["parents"][name] = ledger.entry_tree(name)

    requested = config.get("requested_edges", [])
    if requested:
        source_ids = config["parents"]["hand"]
        leaves = ledger.resolve("hand")
        feasible = feasible_rollup([leaf.histogram for leaf in leaves], requested)
        report["manual_request"] = {
            "parent": "hand",
            "source_entries": list(source_ids),
            "leaf_set_digest": ledger.current["hand"] and ledger.nodes[ledger.current["hand"]].leaf_set_digest,
            **feasible.to_dict(),
        }
    return ledger, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="按 fixtures 汇总异构延迟直方图并报告分位桶值域")
    parser.add_argument("histograms", help="直方图 fixtures 路径")
    parser.add_argument("rollups", help="上卷配置路径")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 报告")
    args = parser.parse_args(argv)

    ledger, report = build_report(*load_fixtures(args.histograms, args.rollups))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    for name, group in report["groups"].items():
        auto = group["auto"]
        print(
            f"父层 {name}: 总量 {auto['population']} (null {auto['null_count']}), "
            f"网格 {auto['grid']}, 计数 {auto['counts']}"
            + (f", 尾桶 {auto['tail_count']}" if auto["tail_present"] else "")
            + f", 叶集合 {group['leaf_set_digest'][:12]}"
        )
        for item in group.get("quantiles", []):
            if not item["defined"]:
                print(f"  p={item['probability']}: 空总体，无定义")
            else:
                print(
                    f"  p={item['probability']}: 秩 {item['rank']}/{item['population']} "
                    f"命中桶 [{encode_edge(item['lower'])}, {item['upper']})"
                )
    manual = report.get("manual_request")
    if manual:
        print(
            f"手动边界 {manual['target_edges']}: "
            + ("精确" if manual["exact"] else "可行域（不可精确拆桶）")
        )
        for bucket in manual["buckets"]:
            print(
                f"  [{bucket['lower_edge']}, {bucket['upper_edge']}) "
                f"计数 ∈ [{bucket['count_range']['min']}, {bucket['count_range']['max']}]"
            )
        print(
            f"  联合总量 {manual['joint_min_total']}（逐桶下界和 {manual['independent_lower_sum']}，"
            f"逐桶上界和 {manual['independent_upper_sum']}）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
