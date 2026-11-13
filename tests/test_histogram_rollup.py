import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from src.histogram_algebra import (
    Histogram,
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
from src.histogram_errors import (
    BoundsError,
    EmptyPopulationError,
    InvalidHistogramError,
    SourceConflictError,
    UnknownSourceError,
    UnsupportedUnitError,
)
from src.histogram_feasibility import feasible_rollup
from src.histogram_registry import (
    LeafRegistry,
    RollupLedger,
    histogram_digest,
    node_digest,
)


def leaf(histogram_id, edges, counts, *, tail_count=0, tail_present=False, null_count=0, unit="us"):
    return Histogram(
        edges,
        counts,
        tail_count=tail_count,
        tail_present=tail_present,
        null_count=null_count,
        unit=unit,
    )


def fixture_leaf(histogram_id, data):
    return histogram_from_dict({**data, "histogram_id": histogram_id})


class FixtureLoadingTests(unittest.TestCase):
    def test_fixture_shapes_match_documented_example(self):
        root = Path(__file__).resolve().parents[1]
        items = json.loads((root / "fixtures/histograms.json").read_text(encoding="utf-8"))["histograms"]
        by_id = {item["histogram_id"]: fixture_leaf(item["histogram_id"], item) for item in items}
        self.assertEqual(by_id["A"].edges, (Fraction(0), Fraction(10), Fraction(20)))
        self.assertEqual(by_id["A"].counts, (2, 2))
        self.assertEqual(by_id["A"].null_count, 1)
        self.assertEqual(by_id["A"].total(), 4)
        self.assertEqual(by_id["B"].edges, (Fraction(0), Fraction(5), Fraction(20)))
        self.assertEqual(by_id["B"].counts, (1, 3))
        self.assertEqual(by_id["tail"].tail_count, 2)
        self.assertTrue(by_id["tail"].tail_present)
        self.assertEqual(by_id["tail"].to_dict()["edges"], [0, 20, "+inf"])

    def test_null_count_never_enters_quantile_denominator(self):
        a = fixture_leaf("A", json.loads((Path(__file__).resolve().parents[1] / "fixtures/histograms.json").read_text(encoding="utf-8"))["histograms"][0])
        # A has counts 2+2 and null_count 1; denominator must be 4, not 5.
        self.assertEqual(a.total(), 4)
        self.assertEqual(a.null_count, 1)

    def test_tail_source_parses(self):
        tail = fixture_leaf(
            "tail",
            {"unit": "us", "edges": [0, 20, "+inf"], "counts": [0, 2], "null_count": 0},
        )
        self.assertEqual(tail.edges, (Fraction(0), Fraction(20)))
        self.assertEqual(tail.counts, (0,))
        self.assertEqual(tail.tail_count, 2)
        self.assertEqual(tail.total(), 2)


class CommonCoarseningTests(unittest.TestCase):
    def setUp(self):
        self.A = fixture_leaf("A", {"unit": "us", "edges": [0, 10, 20], "counts": [2, 2], "null_count": 1})
        self.B = fixture_leaf("B", {"unit": "us", "edges": [0, 5, 20], "counts": [1, 3]})

    def test_common_coarsening_is_the_shared_subset(self):
        # A ∩ B = {0, 20}. 10 is A-only, 5 is B-only.
        self.assertEqual(common_coarsening([self.A, self.B]), (Fraction(0), Fraction(20)))

    def test_auto_rollup_matches_documented_exact_result(self):
        merged = auto_rollup([self.A, self.B])
        self.assertEqual(merged.edges, (Fraction(0), Fraction(20)))
        self.assertEqual(merged.counts, (8,))
        self.assertEqual(merged.total(), 8)
        self.assertEqual(merged.null_count, 1)

    def test_auto_rollup_is_commutative_and_order_stable(self):
        forward = auto_rollup([self.A, self.B])
        reverse = auto_rollup([self.B, self.A])
        self.assertEqual(forward.to_dict(), reverse.to_dict())

    def test_auto_rollup_conserves_integer_total(self):
        self.assertEqual(auto_rollup([self.A, self.B]).total(), self.A.total() + self.B.total())


class ZeroBucketSplitTests(unittest.TestCase):
    def test_zero_bucket_can_be_split_without_interpolation(self):
        source = leaf("z", [0, 10, 20], [0, 4])
        exact = coarsen(source, [0, 5, 10, 20])
        self.assertEqual(exact.edges, (Fraction(0), Fraction(5), Fraction(10), Fraction(20)))
        self.assertEqual(exact.counts, (0, 0, 4))
        self.assertEqual(exact.total(), 4)

    def test_nonempty_bucket_split_is_rejected(self):
        source = leaf("n", [0, 10, 20], [2, 2])
        with self.assertRaises(BoundsError):
            coarsen(source, [0, 5, 10, 20])
        # And the exact request at existing boundaries works.
        self.assertEqual(coarsen(source, [0, 10, 20]).counts, (2, 2))

    def test_manual_rollup_refuses_target_that_cuts_nonempty_source(self):
        A = fixture_leaf("A", {"unit": "us", "edges": [0, 10, 20], "counts": [2, 2]})
        B = fixture_leaf("B", {"unit": "us", "edges": [0, 5, 20], "counts": [1, 3]})
        with self.assertRaises(BoundsError):
            manual_rollup([A, B], [0, 10, 20])

    def test_manual_rollup_exact_when_target_is_coarsening(self):
        A = fixture_leaf("A", {"unit": "us", "edges": [0, 10, 20], "counts": [2, 2]})
        B = fixture_leaf("B", {"unit": "us", "edges": [0, 5, 20], "counts": [1, 3]})
        exact = manual_rollup([A, B], [0, 20])
        self.assertEqual(exact.counts, (8,))

    def test_boundary_below_support_is_allowed_when_empty(self):
        source = leaf("s", [10, 20, 30], [1, 1])
        exact = coarsen(source, [0, 10, 20, 30])
        self.assertEqual(exact.counts, (0, 1, 1))

    def test_dropping_nonzero_data_below_target_is_rejected(self):
        source = leaf("s", [10, 20, 30], [1, 1])
        with self.assertRaises(BoundsError):
            coarsen(source, [20, 30])


class QuantileRangeTests(unittest.TestCase):
    def test_rank_definition_is_ceil_pN_and_a_alone(self):
        A = fixture_leaf("A", {"unit": "us", "edges": [0, 10, 20], "counts": [2, 2]})
        q = quantile_range(A, 0.5)
        # N = 4 => ceil(2.0) = 2; cumulative 2 is first reached in [0, 10).
        self.assertEqual(q.rank, 2)
        self.assertEqual(q.population, 4)
        self.assertEqual((q.lower, q.upper), (Fraction(0), Fraction(10)))
        self.assertEqual(q.bucket_count, 2)
        self.assertEqual(q.to_dict()["lower"], 0)
        self.assertEqual(q.to_dict()["upper"], 10)

    def test_half_open_interval_upper_is_not_inclusive(self):
        A = fixture_leaf("A", {"unit": "us", "edges": [0, 10, 20], "counts": [2, 2]})
        report = quantile_report(A, [0.5])[0]
        self.assertEqual(report["definition"], "smallest bucket whose cumulative count first reaches ceil(p*N)")
        self.assertEqual(report["note"], "bucket bound, not a point estimate")

    def test_infinite_tail_upper_stays_infinite(self):
        tail = fixture_leaf("tail", {"unit": "us", "edges": [0, 20, "+inf"], "counts": [0, 2]})
        q = quantile_range(tail, 0.99)
        self.assertIsNone(q.upper)
        self.assertEqual(q.to_dict()["upper"], "+inf")
        self.assertTrue(q.to_dict()["upper_is_infinite"])

    def test_tail_not_disguised_as_finite(self):
        tail = fixture_leaf("tail", {"unit": "us", "edges": [0, 20, "+inf"], "counts": [0, 2]})
        report = quantile_report(tail, [1.0])[0]
        self.assertEqual(report["upper"], "+inf")
        self.assertTrue(report["upper_is_infinite"])

    def test_ceil_rank_uses_exact_rational_arithmetic(self):
        # 0.1 * 10 = 1 exactly; float arithmetic would also give 1, but 0.7*10=7.000000000000001
        # in binary float would make ceil wrong. Build a direct check on the helper.
        from src.histogram_algebra import ceil_rank

        self.assertEqual(ceil_rank(0.7, 10), 7)
        self.assertEqual(ceil_rank(0.29, 100), 29)
        self.assertEqual(ceil_rank(0.0, 10), 1)
        self.assertEqual(ceil_rank(1.0, 10), 10)

    def test_empty_population_is_undefined_not_nan(self):
        only_null = leaf("n", [0, 10], [0], null_count=5)
        with self.assertRaises(EmptyPopulationError):
            quantile_range(only_null, 0.5)
        report = quantile_report(only_null, [0.5, 0.99])
        self.assertEqual(report[0]["defined"], False)
        self.assertEqual(report[0]["reason"], "empty_population")
        self.assertEqual(report[0]["population"], 0)
        self.assertIsNone(report[0]["upper"])

    def test_quantiles_are_never_averaged_across_leaves(self):
        # Two leaves whose per-leaf p50 buckets differ; pooled p50 must use pooled counts.
        left = leaf("left", [0, 10, 100], [1, 0])
        right = leaf("right", [0, 10, 100], [0, 9])
        pooled = auto_rollup([left, right])
        self.assertEqual(pooled.edges, (Fraction(0), Fraction(10), Fraction(100)))
        self.assertEqual(pooled.counts, (1, 9))
        q = quantile_range(pooled, 0.5)
        self.assertEqual(q.rank, 5)
        self.assertEqual((q.lower, q.upper), (Fraction(10), Fraction(100)))
        # Naive mean of per-leaf p50 buckets would wrongly land in [0, 10).
        self.assertNotEqual((q.lower, q.upper), (Fraction(0), Fraction(10)))


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.A = fixture_leaf("A", {"unit": "us", "edges": [0, 10, 20], "counts": [2, 2]})
        self.B = fixture_leaf("B", {"unit": "us", "edges": [0, 5, 20], "counts": [1, 3]})

    def test_target_10_splits_B_and_yields_ranges_not_exact_counts(self):
        result = feasible_rollup([self.A, self.B], [0, 10, 20])
        self.assertFalse(result.exact)
        self.assertIn(Fraction(10), result.unsplittable_edges)
        self.assertEqual(result.buckets[0].to_dict()["min"], 3)
        self.assertEqual(result.buckets[0].to_dict()["max"], 6)
        self.assertEqual(result.buckets[1].to_dict()["min"], 2)
        self.assertEqual(result.buckets[1].to_dict()["max"], 5)

    def test_joint_total_is_exact_but_independent_sums_drift(self):
        result = feasible_rollup([self.A, self.B], [0, 10, 20])
        self.assertEqual(result.joint_min_total, 8)
        self.assertEqual(result.joint_max_total, 8)
        self.assertEqual(result.population, 8)
        self.assertEqual(sum(result.witness), 8)
        self.assertLess(result.independent_lower_sum, 8)
        self.assertGreater(result.independent_upper_sum, 8)

    def test_witness_lands_inside_every_bucket_range(self):
        result = feasible_rollup([self.A, self.B], [0, 10, 20])
        for index, bucket in enumerate(result.buckets):
            self.assertGreaterEqual(result.witness[index], bucket.lower)
            self.assertLessEqual(result.witness[index], bucket.upper)

    def test_exact_request_reports_unique_counts(self):
        result = feasible_rollup([self.A, self.B], [0, 20])
        self.assertTrue(result.exact)
        self.assertEqual(result.buckets[0].lower, 8)
        self.assertEqual(result.buckets[0].upper, 8)
        self.assertEqual(result.independent_lower_sum, 8)
        self.assertEqual(result.independent_upper_sum, 8)

    def test_zero_count_bucket_can_still_be_cut_feasibly(self):
        zero = leaf("z", [0, 10, 20], [0, 4])
        result = feasible_rollup([zero], [0, 5, 10, 20])
        self.assertTrue(result.exact)
        # 目标网格 [0,5),[5,10),[10,20),[20,+inf)；源没有尾桶，故末桶为 0。
        self.assertEqual([b.lower for b in result.buckets], [0, 0, 4, 0])

    def test_infeasible_target_below_nonzero_data_rejected(self):
        with self.assertRaises(BoundsError):
            feasible_rollup([self.A], [5, 10, 20])


class UnitTests(unittest.TestCase):
    def test_mixed_units_use_exact_rational_conversion(self):
        us = leaf("us", [0, 1000, 2000], [1, 1], unit="us")
        ms = leaf("ms", [0, 1, 2], [1, 1], unit="ms")
        normalized = normalize_units([us, ms], "us")
        self.assertEqual(normalized[0].edges, (Fraction(0), Fraction(1000), Fraction(2000)))
        self.assertEqual(normalized[1].edges, (Fraction(0), Fraction(1000), Fraction(2000)))
        merged = auto_rollup([us, ms], "us")
        self.assertEqual(merged.edges, (Fraction(0), Fraction(1000), Fraction(2000)))
        self.assertEqual(merged.counts, (2, 2))
        self.assertEqual(merged.total(), 4)

    def test_unregistered_unit_is_rejected(self):
        weird = leaf("w", [0, 1], [1], unit="furlong")
        with self.assertRaises(UnsupportedUnitError):
            normalize_units([weird], "us")

    def test_conversion_preserves_counts_integers(self):
        # Counts are not rescaled; only edges are. Integer totals are preserved.
        ns = leaf("ns", [0, 500000, 1_000_000], [3, 4], unit="ns")
        merged = auto_rollup([ns], "us")
        self.assertEqual(merged.edges, (Fraction(0), Fraction(500), Fraction(1000)))
        self.assertEqual(merged.counts, (3, 4))
        self.assertEqual(merged.total(), 7)


class TailHandlingTests(unittest.TestCase):
    def test_auto_rollup_preserves_explicit_tail(self):
        A = fixture_leaf("A", {"unit": "us", "edges": [0, 10, 20], "counts": [2, 2]})
        tail = fixture_leaf("tail", {"unit": "us", "edges": [0, 20, "+inf"], "counts": [0, 2]})
        merged = auto_rollup([A, tail])
        self.assertEqual(merged.edges, (Fraction(0), Fraction(20)))
        self.assertEqual(merged.counts, (4,))
        self.assertEqual(merged.tail_count, 2)
        self.assertTrue(merged.tail_present)
        self.assertEqual(merged.total(), 6)

    def test_zero_tail_keeps_infinite_label_when_source_declared_it(self):
        source = leaf("t", [0, 10, 20], [2, 2], tail_count=0, tail_present=True)
        self.assertTrue(source.to_dict()["edges"][-1] == "+inf")
        self.assertEqual(source.to_dict()["counts"][-1], 0)

    def test_tail_count_enters_denominator_but_null_does_not(self):
        source = leaf("t", [0, 10], [1], tail_count=3, tail_present=True, null_count=7)
        self.assertEqual(source.total(), 4)
        q = quantile_range(source, 1.0)
        self.assertEqual(q.rank, 4)
        self.assertIsNone(q.upper)


class RegistryTests(unittest.TestCase):
    def test_latest_revision_replaces_not_accumulates(self):
        registry = LeafRegistry()
        registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2]))
        result = registry.register("A", 2, leaf("A", [0, 10, 20], [5, 5]))
        self.assertEqual(result["status"], "replaced")
        self.assertEqual(registry.latest("A").revision, 2)
        self.assertEqual(registry.latest("A").histogram.total(), 10)

    def test_same_revision_different_content_conflicts(self):
        registry = LeafRegistry()
        registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2]))
        with self.assertRaises(SourceConflictError):
            registry.register("A", 1, leaf("A", [0, 10, 20], [3, 3]))

    def test_same_revision_same_content_is_idempotent(self):
        registry = LeafRegistry()
        registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2]))
        result = registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2]))
        self.assertEqual(result["status"], "unchanged")

    def test_stale_revision_does_not_override(self):
        registry = LeafRegistry()
        registry.register("A", 2, leaf("A", [0, 10, 20], [5, 5]))
        result = registry.register("A", 1, leaf("A", [0, 10, 20], [1, 1]))
        self.assertEqual(result["status"], "stale")
        self.assertEqual(registry.latest("A").revision, 2)
        self.assertEqual(registry.latest("A").histogram.total(), 10)

    def test_unknown_source_raises(self):
        registry = LeafRegistry()
        with self.assertRaises(UnknownSourceError):
            registry.latest("missing")

    def test_persistence_round_trip_keeps_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaves.json"
            registry = LeafRegistry()
            registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2], null_count=1))
            registry.register("B", 1, leaf("B", [0, 5, 20], [1, 3]))
            registry.save(path)
            restored = LeafRegistry(path)
            self.assertEqual(restored.revisions(), {"A": 1, "B": 1})
            self.assertEqual(restored.latest("A").digest, registry.latest("A").digest)
            self.assertEqual(restored.latest("A").histogram.to_dict(), registry.latest("A").histogram.to_dict())


class LedgerTests(unittest.TestCase):
    def _ledger(self):
        registry = LeafRegistry()
        registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2], null_count=1))
        registry.register("B", 1, leaf("B", [0, 5, 20], [1, 3]))
        registry.register("tail", 1, leaf("tail", [0, 20], [0], tail_count=2, tail_present=True))
        ledger = RollupLedger(registry)
        ledger.define_parent("hand", ["A", "B"])
        ledger.define_parent("all", ["A", "B", "tail"])
        return ledger

    def test_parent_version_determined_by_leaf_set_not_order(self):
        ledger = self._ledger()
        first = ledger.build("hand")
        ledger.define_parent("hand_reordered", ["B", "A"])
        second = ledger.build("hand_reordered")
        self.assertEqual(first.node_id, second.node_id)
        self.assertEqual(first.leaf_set_digest, second.leaf_set_digest)
        self.assertEqual(first.histogram.to_dict(), second.histogram.to_dict())

    def test_revision_revokes_old_contribution_then_adds_new(self):
        ledger = self._ledger()
        before = ledger.build("hand")
        self.assertEqual(before.histogram.total(), 8)
        report = ledger.apply_revision("A", 2, leaf("A", [0, 10, 20], [4, 4], null_count=1))
        self.assertEqual(len(report["rebuilds"]), 1)
        build = report["rebuilds"][0]
        self.assertEqual(build["revoked"]["revision"], 1)
        self.assertEqual(build["revoked"]["population_removed"], 4)
        self.assertEqual(build["added"]["population_added"], 8)
        self.assertEqual(build["parent_population_before"], 8)
        self.assertEqual(build["parent_population_after"], 12)
        after = ledger.nodes[ledger.current["hand"]]
        self.assertEqual(after.histogram.total(), 12)
        self.assertEqual(after.histogram.null_count, 1)
        ops = [event["op"] for event in ledger.events]
        self.assertIn("revoke-contribution", ops)
        self.assertIn("add-contribution", ops)

    def test_revision_rebuilds_each_parent_containing_the_leaf(self):
        ledger = self._ledger()
        ledger.build("hand")
        ledger.build("all")
        report = ledger.apply_revision("B", 2, leaf("B", [0, 5, 20], [2, 6]))
        self.assertEqual({rebuild["parent"] for rebuild in report["rebuilds"]}, {"hand", "all"})
        self.assertEqual(ledger.nodes[ledger.current["all"]].histogram.total(), 2 + 4 + 6 + 2)

    def test_stale_revision_does_not_touch_parents(self):
        ledger = self._ledger()
        ledger.build("hand")
        before = ledger.current["hand"]
        report = ledger.apply_revision("A", 0, leaf("A", [0, 10, 20], [1, 1]))
        self.assertEqual(report["rebuilds"], [])
        self.assertEqual(ledger.current["hand"], before)

    def test_duplicate_leaf_entry_rejected(self):
        registry = LeafRegistry()
        registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2]))
        ledger = RollupLedger(registry)
        ledger.define_parent("dup", ["A", "A"])
        with self.assertRaises(SourceConflictError):
            ledger.resolve("dup")

    def test_leaf_cannot_enter_via_two_parent_entries(self):
        registry = LeafRegistry()
        registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2]))
        registry.register("B", 1, leaf("B", [0, 5, 20], [1, 3]))
        ledger = RollupLedger(registry)
        ledger.define_parent("left", ["A"])
        ledger.define_parent("right", ["A", "B"])
        ledger.define_parent("combo", ["left", "right"])
        with self.assertRaises(SourceConflictError):
            ledger.resolve("combo")
        # Independent paths without overlap still work.
        ledger.define_parent("left_only", ["A"])
        ledger.define_parent("right_only", ["B"])
        ledger.define_parent("combo_ok", ["left_only", "right_only"])
        self.assertEqual({leaf.histogram_id for leaf in ledger.resolve("combo_ok")}, {"A", "B"})

    def test_cycle_rejected(self):
        registry = LeafRegistry()
        registry.register("A", 1, leaf("A", [0, 10, 20], [2, 2]))
        ledger = RollupLedger(registry)
        ledger.define_parent("p1", ["A"])
        ledger.define_parent("p2", ["p1"])
        ledger.parents["p1"] = ("p2",)
        with self.assertRaises(SourceConflictError):
            ledger.resolve("p1")

    def test_merge_tree_is_traceable_and_persistable(self):
        ledger = self._ledger()
        ledger.build("hand")
        ledger.build("all")
        summary = ledger.current_summary()
        self.assertIn("hand", summary["parents"])
        self.assertIn("all", summary["parents"])
        self.assertEqual(summary["parents"]["hand"]["histogram"]["counts"], [8])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.json"
            ledger.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data["parents"]), {"hand", "all"})
            self.assertEqual(len(data["nodes"]), 2)
            restored = RollupLedger(LeafRegistry())
            restored.load_state(path, ledger.registry)
            self.assertEqual(restored.current, ledger.current)

    def test_leaf_set_summary_reports_revisions(self):
        ledger = self._ledger()
        ledger.build("hand")
        node = ledger.nodes[ledger.current["hand"]]
        self.assertEqual(node.leaf_ids, ("A", "B"))
        self.assertEqual([leaf["revision"] for leaf in node.leaves], [1, 1])
        self.assertEqual(node.histogram.total(), 8)


class MergeAndEdgeCaseTests(unittest.TestCase):
    def test_merge_requires_same_grid(self):
        left = leaf("l", [0, 10, 20], [1, 1])
        right = leaf("r", [0, 5, 20], [1, 1])
        with self.assertRaises(Exception):
            merge_histograms([left, right])

    def test_empty_group_rejected(self):
        with self.assertRaises(Exception):
            auto_rollup([])

    def test_single_leaf_auto_rollup_keeps_its_grid(self):
        only = leaf("only", [0, 7, 14], [1, 2])
        merged = auto_rollup([only])
        self.assertEqual(merged.counts, (1, 2))
        self.assertEqual(merged.total(), 3)

    def test_quantile_report_reports_every_probability(self):
        source = leaf("q", [0, 10, 20], [1, 3])
        report = quantile_report(source, [0.25, 0.5, 0.75, 0.99])
        self.assertEqual(len(report), 4)
        self.assertTrue(all(item["defined"] for item in report))
        self.assertEqual(report[0]["rank"], 1)
        self.assertEqual(report[-1]["rank"], 4)

    def test_encode_never_emits_infinity_float(self):
        for value in (Fraction(0), Fraction(5, 2)):
            self.assertNotEqual(encode_edge(value), float("inf"))
        self.assertEqual(encode_edges([Fraction(0), Fraction(10)]), [0, 10])

    def test_invalid_counts_rejected(self):
        with self.assertRaises(InvalidHistogramError):
            leaf("bad", [0, 10], [1, 2])
        with self.assertRaises(InvalidHistogramError):
            leaf("bad", [10, 0], [1])
        with self.assertRaises(InvalidHistogramError):
            leaf("bad", [0, 10], [-1])


if __name__ == "__main__":
    unittest.main()
