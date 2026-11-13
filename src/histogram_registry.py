"""持久化叶直方图登记、版本化上卷与可追溯合并树。

身份与替换
==========

* 叶的身份是 ``histogram_id``；``revision`` 是**替换**语义，不是可重复相加的
  标签。同一 id 的新 revision 撤销旧贡献、加入新贡献，绝不重复累加。
* 同一 ``(id, revision)`` 再次登记时内容摘要必须一致：一致则幂等返回，
  不一致抛 ``SourceConflictError``；更低的 revision 视为过期，既不覆盖高
  revision，也不进入任何父层。
* 父层条目可以是叶 id，也可以是另一个父层（组合父层），构建时递归展开为
  叶版本集合。**同一个叶来源只能通过一个入口进入**：若两个入口展开出同一
  叶 id（无论 revision 是否相同），构建该父层即抛 ``SourceConflictError``，
  不做静默去重。

父层版本与合并树
================

* 父层版本由叶版本集合 ``(id, revision, digest)`` 的排序结果决定，与构建顺序
  无关；再叠加模式（auto/manual）、单位与目标网格做内容寻址，得到 ``node_id``。
* 每个节点记录叶集合摘要、结果直方图、子入口与父层名，可回溯撤销/新增事件。
* 修订到达时，先移除旧叶版本的贡献，再加入新版本的贡献，并校验父层总量
  与 ``null_count`` 的加减恒等式。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .histogram_algebra import (
    Histogram,
    auto_rollup,
    encode_edge,
    histogram_from_dict,
    manual_rollup,
)
from .histogram_errors import (
    HistogramError,
    InvalidHistogramError,
    SourceConflictError,
    UnknownSourceError,
)


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def histogram_digest(histogram: Histogram) -> str:
    return hashlib.sha256(canonical_json(histogram.to_dict()).encode("utf-8")).hexdigest()


def node_digest(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _require_revision(revision):
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise InvalidHistogramError("revision 必须是非负整数")
    return revision


@dataclass(frozen=True)
class LeafVersion:
    histogram_id: str
    revision: int
    histogram: Histogram
    digest: str

    def summary(self) -> dict:
        return {
            "histogram_id": self.histogram_id,
            "revision": self.revision,
            "digest": self.digest,
            "unit": self.histogram.unit,
            "population": self.histogram.total(),
            "null_count": self.histogram.null_count,
            "edges": [encode_edge(edge) for edge in self.histogram.edges],
            "tail_present": self.histogram.tail_present,
        }


# ---------------------------------------------------------------------------
# 叶登记
# ---------------------------------------------------------------------------

class LeafRegistry:
    """每个 ``histogram_id`` 只保留最高 revision 的持久登记表。"""

    def __init__(self, path: Optional[Path] = None, *, unit: Optional[str] = None):
        self.path = Path(path) if path is not None else None
        self.unit = unit
        self._leaves: Dict[str, LeafVersion] = {}
        self.events: List[dict] = []
        if self.path is not None and self.path.exists():
            self.load()

    # ---- 查询 -------------------------------------------------------
    def latest(self, histogram_id: str) -> LeafVersion:
        try:
            return self._leaves[histogram_id]
        except KeyError:
            raise UnknownSourceError(f"未登记的叶：{histogram_id}") from None

    def latest_or_none(self, histogram_id: str) -> Optional[LeafVersion]:
        return self._leaves.get(histogram_id)

    def __contains__(self, histogram_id: str) -> bool:
        return histogram_id in self._leaves

    def revisions(self) -> Dict[str, int]:
        return {key: leaf.revision for key, leaf in self._leaves.items()}

    def leaf_set_summary(self, ids: Optional[Sequence[str]] = None) -> List[dict]:
        if ids is None:
            selected = list(self._leaves.values())
        else:
            selected = [self.latest(leaf_id) for leaf_id in ids]
        selected.sort(key=lambda leaf: leaf.histogram_id)
        return [leaf.summary() for leaf in selected]

    def as_dict(self) -> dict:
        return {
            "version": 1,
            "unit": self.unit,
            "leaves": [
                {
                    "histogram_id": leaf.histogram_id,
                    "revision": leaf.revision,
                    "digest": leaf.digest,
                    "histogram": {**leaf.histogram.to_dict(), "unit": leaf.histogram.unit},
                }
                for leaf in sorted(self._leaves.values(), key=lambda leaf: leaf.histogram_id)
            ],
        }

    # ---- 登记 -------------------------------------------------------
    def register(self, histogram_id: str, revision, histogram: Histogram) -> dict:
        if not isinstance(histogram_id, str) or not histogram_id:
            raise InvalidHistogramError("histogram_id 必须是非空字符串")
        if not isinstance(histogram, Histogram):
            raise InvalidHistogramError("只能登记 Histogram 实例")
        revision = _require_revision(revision)
        digest = histogram_digest(histogram)
        current = self._leaves.get(histogram_id)

        if current is None:
            self._leaves[histogram_id] = LeafVersion(histogram_id, revision, histogram, digest)
            event = self._record("add", histogram_id, revision, digest, {})
            return {"status": "added", "event": event, "leaf": self._leaves[histogram_id].summary()}

        if revision == current.revision:
            if digest != current.digest:
                raise SourceConflictError(
                    f"{histogram_id} revision {revision} 已以不同内容登记（摘要 {current.digest[:12]}）"
                )
            return {"status": "unchanged", "leaf": current.summary()}

        if revision < current.revision:
            event = self._record(
                "stale-ignored", histogram_id, revision, digest,
                {"current_revision": current.revision, "current_digest": current.digest},
            )
            return {"status": "stale", "event": event, "leaf": current.summary()}

        self._leaves[histogram_id] = LeafVersion(histogram_id, revision, histogram, digest)
        event = self._record(
            "replace", histogram_id, revision, digest,
            {"previous_revision": current.revision, "previous_digest": current.digest},
        )
        return {"status": "replaced", "event": event, "leaf": self._leaves[histogram_id].summary()}

    def _record(self, op, histogram_id, revision, digest, extra):
        event = {"op": op, "histogram_id": histogram_id, "revision": revision, "digest": digest, **extra}
        self.events.append(event)
        return event

    # ---- 持久化 -----------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise HistogramError("未指定登记表路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def load(self, path: Optional[Path] = None) -> "LeafRegistry":
        target = Path(path) if path is not None else self.path
        if target is None:
            raise HistogramError("未指定登记表路径")
        if not target.exists():
            raise FileNotFoundError(target)
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("leaves"), list):
            raise InvalidHistogramError("登记表必须是包含 leaves 数组的对象")
        leaves: Dict[str, LeafVersion] = {}
        for item in data["leaves"]:
            if not isinstance(item, dict):
                raise InvalidHistogramError("登记表中的叶必须是对象")
            histogram_id = item.get("histogram_id")
            revision = _require_revision(item.get("revision"))
            histogram = histogram_from_dict(item.get("histogram", {}))
            digest = histogram_digest(histogram)
            if item.get("digest") not in (None, digest):
                raise SourceConflictError(f"{histogram_id} 的登记摘要与实际内容不一致")
            if histogram_id in leaves:
                raise SourceConflictError(f"登记表包含重复的叶 {histogram_id}")
            leaves[histogram_id] = LeafVersion(histogram_id, revision, histogram, digest)
        self._leaves = leaves
        self.unit = data.get("unit", self.unit)
        self.path = target
        return self


# ---------------------------------------------------------------------------
# 版本化上卷
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollupNode:
    node_id: str
    parent: str
    mode: str
    leaves: Tuple[dict, ...]
    grid: Tuple
    unit: str
    histogram: Histogram
    entries: Tuple[str, ...]
    requested_edges: Optional[Tuple] = None

    @property
    def leaf_set_digest(self) -> str:
        """父层版本只由叶版本集合决定，与构建顺序无关。"""
        return node_digest(
            [[leaf["histogram_id"], leaf["revision"], leaf["digest"]] for leaf in self.leaves]
        )

    @property
    def leaf_ids(self) -> Tuple[str, ...]:
        return tuple(leaf["histogram_id"] for leaf in self.leaves)

    def summary(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent": self.parent,
            "mode": self.mode,
            "entries": list(self.entries),
            "leaf_versions": [
                {"histogram_id": leaf["histogram_id"], "revision": leaf["revision"], "digest": leaf["digest"]}
                for leaf in self.leaves
            ],
            "leaf_set_digest": self.leaf_set_digest,
            "grid": [encode_edge(edge) for edge in self.grid],
            "unit": self.unit,
            "population": self.histogram.total(),
            "null_count": self.histogram.null_count,
            "requested_edges": None if self.requested_edges is None else [encode_edge(e) for e in self.requested_edges],
        }


class RollupLedger:
    """父层定义、当前节点与修订事件；全部整数守恒。"""

    def __init__(self, registry: LeafRegistry):
        self.registry = registry
        self.parents: Dict[str, Tuple[str, ...]] = {}
        self.nodes: Dict[str, RollupNode] = {}
        self.current: Dict[str, str] = {}
        self.events: List[dict] = []

    # ---- 定义 -------------------------------------------------------
    def define_parent(self, name: str, entries: Sequence[str]) -> dict:
        """声明父层。条目可以是叶 id 或另一个父层名，且不能自引用。"""
        if not isinstance(name, str) or not name:
            raise InvalidHistogramError("父层名必须是非空字符串")
        if not isinstance(entries, (list, tuple)) or not entries:
            raise InvalidHistogramError("父层必须至少包含一个条目")
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise InvalidHistogramError("父层条目必须是非空字符串")
            if entry == name:
                raise SourceConflictError(f"父层 {name} 不能把自身作为条目")
        if name in self.parents:
            raise SourceConflictError(f"父层 {name} 已定义")
        self.parents[name] = tuple(entries)
        self.events.append({"op": "define-parent", "parent": name, "entries": list(entries)})
        return {"parent": name, "entries": list(entries)}

    def _expand(self, entry: str, *, stack=()) -> List[LeafVersion]:
        """把一个父层条目递归展开为叶版本；同一叶 id 只能出现一次。"""
        if entry in self.parents:
            if entry in stack:
                raise SourceConflictError(f"父层定义存在循环：{' -> '.join(stack + (entry,))}")
            collected: List[LeafVersion] = []
            for child in self.parents[entry]:
                collected.extend(self._expand(child, stack=stack + (entry,)))
            return collected
        return [self.registry.latest(entry)]

    def resolve(self, parent: str) -> List[LeafVersion]:
        """返回该父层的叶版本集合；两个入口解析出同一叶 id 时抛冲突。"""
        if parent not in self.parents:
            raise UnknownSourceError(f"未定义的父层：{parent}")
        leaves = self._expand(parent)
        seen: Dict[str, LeafVersion] = {}
        origins: Dict[str, str] = {}
        for leaf in leaves:
            if leaf.histogram_id in seen:
                raise SourceConflictError(
                    f"叶来源 {leaf.histogram_id} 从两个父入口重复进入父层 {parent}"
                )
            seen[leaf.histogram_id] = leaf
        return list(seen.values())

    def entry_tree(self, parent: str) -> dict:
        """父层入口结构，用于说明叶来源只经由一个入口进入。"""
        def walk(entry, stack):
            if entry in self.parents:
                return {
                    "parent": entry,
                    "children": [walk(child, stack + (entry,)) for child in self.parents[entry]],
                }
            leaf = self.registry.latest(entry)
            return {"leaf": entry, "revision": leaf.revision, "digest": leaf.digest}

        return walk(parent, ())

    # ---- 构建 -------------------------------------------------------
    def build(
        self,
        parent: str,
        *,
        unit: Optional[str] = None,
        mode: str = "auto",
        requested_edges: Optional[Sequence] = None,
    ) -> RollupNode:
        leaves = self.resolve(parent)
        if not leaves:
            raise HistogramError("父层没有可汇总的叶")
        histograms = [leaf.histogram for leaf in leaves]
        if mode == "auto":
            result = auto_rollup(histograms, unit)
            normalized_requested = None
        elif mode == "manual":
            if requested_edges is None:
                raise HistogramError("手动模式需要目标边界")
            result = manual_rollup(histograms, requested_edges, unit)
            normalized_requested = tuple(requested_edges)
        else:
            raise HistogramError(f"未知的上卷模式：{mode}")

        summaries = tuple(
            leaf.summary() for leaf in sorted(leaves, key=lambda leaf: leaf.histogram_id)
        )
        payload = {
            # 父层版本只由叶版本集合、模式、单位与网格决定；父层名不参与寻址，
            # 因此同一叶集合经不同入口命名仍指向同一节点。
            "mode": mode,
            "unit": result.unit,
            "grid": [encode_edge(edge) for edge in result.edges],
            "tail_present": result.tail_present,
            "requested_edges": None if normalized_requested is None else [encode_edge(e) for e in normalized_requested],
            "leaf_versions": [[leaf["histogram_id"], leaf["revision"], leaf["digest"]] for leaf in summaries],
        }
        node_id = node_digest(payload)

        node = self.nodes.get(node_id)
        if node is None:
            node = RollupNode(
                node_id=node_id,
                parent=parent,
                mode=mode,
                leaves=summaries,
                grid=tuple(result.edges),
                unit=result.unit,
                histogram=result,
                entries=self.parents[parent],
                requested_edges=normalized_requested,
            )
            self.nodes[node_id] = node
            self.events.append({"op": "add-node", "parent": parent, "node_id": node_id, "mode": mode})
        previous = self.current.get(parent)
        self.current[parent] = node_id
        if previous is not None and previous != node_id:
            self.events.append({"op": "supersede-node", "parent": parent, "revoked": previous, "current": node_id})
        return node

    def apply_revision(self, histogram_id: str, revision, histogram: Histogram, *, unit: Optional[str] = None) -> dict:
        """登记新 revision，并对每个包含该叶的父层撤销旧贡献、加入新贡献。"""
        old = self.registry.latest_or_none(histogram_id)
        affected = [name for name in self.parents if histogram_id in {leaf.histogram_id for leaf in self.resolve(name)}]
        before = {name: self.nodes[self.current[name]] for name in affected if name in self.current}

        registration = self.registry.register(histogram_id, revision, histogram)
        if registration["status"] in ("stale", "unchanged"):
            return {"registration": registration, "rebuilds": []}

        rebuilds: List[dict] = []
        for name in affected:
            previous_node = before.get(name)
            rebuilt = self.build(
                name,
                unit=unit,
                mode=previous_node.mode if previous_node else "auto",
                requested_edges=previous_node.requested_edges if previous_node else None,
            )
            if previous_node is None:
                continue
            old_population = old.histogram.total() if old else 0
            old_nulls = old.histogram.null_count if old else 0
            expected_population = previous_node.histogram.total() - old_population + histogram.total()
            expected_nulls = previous_node.histogram.null_count - old_nulls + histogram.null_count
            if rebuilt.histogram.total() != expected_population:
                raise HistogramError(
                    f"父层 {name} 修订后总量 {rebuilt.histogram.total()} != 预期 {expected_population}"
                )
            if rebuilt.histogram.null_count != expected_nulls:
                raise HistogramError(f"父层 {name} 修订后 null_count 不守恒")
            rebuilds.append(
                {
                    "parent": name,
                    "revoked": {
                        "histogram_id": histogram_id,
                        "revision": old.revision if old else None,
                        "population_removed": old_population,
                        "null_count_removed": old_nulls,
                        "node_id": previous_node.node_id,
                    },
                    "added": {
                        "histogram_id": histogram_id,
                        "revision": revision,
                        "population_added": histogram.total(),
                        "null_count_added": histogram.null_count,
                        "node_id": rebuilt.node_id,
                    },
                    "parent_population_before": previous_node.histogram.total(),
                    "parent_population_after": rebuilt.histogram.total(),
                }
            )
            self.events.append(
                {"op": "revoke-contribution", "parent": name, "histogram_id": histogram_id,
                 "revision": old.revision if old else None, "node_id": previous_node.node_id}
            )
            self.events.append(
                {"op": "add-contribution", "parent": name, "histogram_id": histogram_id,
                 "revision": revision, "node_id": rebuilt.node_id}
            )
        return {"registration": registration, "rebuilds": rebuilds}

    # ---- 视图与持久化 -----------------------------------------------
    def current_summary(self) -> dict:
        parents = {}
        for name, node_id in sorted(self.current.items()):
            node = self.nodes[node_id]
            parents[name] = {**node.summary(), "histogram": node.histogram.to_dict()}
        return {
            "parents": parents,
            "entries": {name: self.entry_tree(name) for name in sorted(self.parents)},
            "leaf_set_digests": {name: self.nodes[node_id].leaf_set_digest for name, node_id in sorted(self.current.items())},
            "events": list(self.events),
        }

    def as_dict(self) -> dict:
        return {
            "version": 1,
            "parents": {name: list(entries) for name, entries in self.parents.items()},
            "current": dict(self.current),
            "nodes": {
                node_id: {
                    "summary": node.summary(),
                    "histogram": node.histogram.to_dict(),
                }
                for node_id, node in sorted(self.nodes.items())
            },
            "events": list(self.events),
        }

    def save(self, path: Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def load_state(self, path: Path, registry: Optional[LeafRegistry] = None) -> "RollupLedger":
        """从磁盘恢复父层定义、节点与当前指针（节点直方图原样读回）。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "parents" not in data:
            raise InvalidHistogramError("合并树状态必须包含 parents")
        self.registry = registry or self.registry
        self.parents = {name: tuple(entries) for name, entries in data["parents"].items()}
        self.current = dict(data.get("current", {}))
        self.nodes = {}
        for node_id, item in data.get("nodes", {}).items():
            hist = histogram_from_dict(item["histogram"])
            summary = item.get("summary", {})
            self.nodes[node_id] = RollupNode(
                node_id=node_id,
                parent=summary.get("parent", ""),
                mode=summary.get("mode", "auto"),
                leaves=tuple(summary.get("leaf_versions", [])),
                grid=tuple(hist.edges),
                unit=hist.unit,
                histogram=hist,
                entries=tuple(summary.get("entries", [])),
                requested_edges=None,
            )
        self.events = list(data.get("events", []))
        return self
