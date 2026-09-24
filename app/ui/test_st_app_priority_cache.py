"""The priority memo, and the one invariant that keeps it honest.

The ML Predictions tab re-scores the whole table on every rerun, and Streamlit
reruns on every widget interaction anywhere on the page. `priority_for_current_frame`
serves the previous answer when nothing that feeds it has moved (D8).

A cache over a frame the rest of the app mutates is exactly the failure mode
`reset_selection_state` already exists to prevent, so the tests below care less
about the hit than about every way the memo is supposed to *miss*.
"""
import datetime
import types
import unittest
from unittest.mock import patch

import pandas as pd

from app.ui import st_app

BALANCED = {"prox": 0.35, "stale": 0.35, "prior": 0.20, "scope": 0.10}
LOCAL = {"prox": 0.55, "stale": 0.25, "prior": 0.15, "scope": 0.05}


def a_frame(n=3):
    return pd.DataFrame([
        {'fhrsid': str(i), 'latitude': 51.4 + i / 100, 'longitude': -0.12,
         'postcode': 'SW16 1AA', 'in_scope': True}
        for i in range(n)
    ])


class CacheTestCase(unittest.TestCase):
    """Each test gets a fresh `session_state`, as a plain namespace-with-dict.

    A `MagicMock` would answer `.get` with a truthy Mock and every miss would
    silently look like a hit.
    """

    def setUp(self):
        self.state = _FakeSessionState()
        self._patch = patch.object(st_app, 'st')
        self.mock_st = self._patch.start()
        self.mock_st.session_state = self.state
        self.addCleanup(self._patch.stop)


class _FakeSessionState(dict):
    """`st.session_state` is both a dict and an attribute bag."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class TestTheMemoReturnsTheSameScores(CacheTestCase):

    def test_a_second_call_with_everything_unchanged_does_not_recompute(self):
        self.state['data_version'] = 1
        df = a_frame()
        with patch.object(st_app, 'calculate_restaurant_priority',
                          wraps=st_app.calculate_restaurant_priority) as spy:
            first = st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
            second = st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        self.assertEqual(spy.call_count, 1)
        pd.testing.assert_frame_equal(first, second)

    def test_the_cached_answer_equals_the_uncached_one(self):
        """The memo must not be the only thing that ever computed the scores."""
        self.state['data_version'] = 1
        df = a_frame()
        st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        cached = st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        direct = st_app.calculate_restaurant_priority(
            df, anchor_lat=51.4212, anchor_lon=-0.1292, weights=BALANCED)
        pd.testing.assert_frame_equal(cached, direct)

    def test_each_caller_gets_its_own_frame(self):
        """Returning the cached object itself would let one caller's mutation
        become the next rerun's answer."""
        self.state['data_version'] = 1
        df = a_frame()
        first = st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        first['priority_score'] = -999.0
        second = st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        self.assertNotIn(-999.0, list(second['priority_score']))


class TestTheMemoMissesWhenItShould(CacheTestCase):

    def _count_calls(self, calls):
        with patch.object(st_app, 'calculate_restaurant_priority',
                          wraps=st_app.calculate_restaurant_priority) as spy:
            for args in calls:
                st_app.priority_for_current_frame(*args)
            return spy.call_count

    def test_a_new_anchor_recomputes(self):
        self.state['data_version'] = 1
        df = a_frame()
        self.assertEqual(self._count_calls([
            (df, 51.4212, -0.1292, BALANCED),
            (df, 51.5074, -0.1278, BALANCED),
        ]), 2)

    def test_a_new_strategy_preset_recomputes(self):
        self.state['data_version'] = 1
        df = a_frame()
        self.assertEqual(self._count_calls([
            (df, 51.4212, -0.1292, BALANCED),
            (df, 51.4212, -0.1292, LOCAL),
        ]), 2)

    def test_a_reloaded_frame_recomputes(self):
        """The one that matters. A triage write reloads the table; serving the
        previous scores would rank restaurants by a scope flag they no longer
        have."""
        df = a_frame()
        st_app.set_enriched_frame(df)
        st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        reloaded = a_frame(5)
        st_app.set_enriched_frame(reloaded)
        with patch.object(st_app, 'calculate_restaurant_priority',
                          wraps=st_app.calculate_restaurant_priority) as spy:
            out = st_app.priority_for_current_frame(reloaded, 51.4212, -0.1292, BALANCED)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(out), 5)

    def test_a_frame_of_the_same_length_still_recomputes_after_a_reload(self):
        """A rating write changes values, not row count. Keying on the shape
        would miss it."""
        df = a_frame()
        st_app.set_enriched_frame(df)
        st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        same_size = a_frame()
        same_size['in_scope'] = False
        st_app.set_enriched_frame(same_size)
        with patch.object(st_app, 'calculate_restaurant_priority',
                          wraps=st_app.calculate_restaurant_priority) as spy:
            out = st_app.priority_for_current_frame(same_size, 51.4212, -0.1292, BALANCED)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(list(out['priority_score']), [0.0, 0.0, 0.0])

    def test_a_new_day_recomputes(self):
        """Staleness is measured against today, so yesterday's scores are
        wrong for a session left open overnight."""
        self.state['data_version'] = 1
        df = a_frame()
        real_date = datetime.date

        class FrozenDate(real_date):
            frozen = real_date(2026, 9, 24)

            @classmethod
            def today(cls):
                return cls.frozen

        with patch.object(st_app.datetime, 'date', FrozenDate), \
             patch.object(st_app, 'calculate_restaurant_priority',
                          wraps=st_app.calculate_restaurant_priority) as spy:
            st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
            FrozenDate.frozen = real_date(2026, 9, 25)
            st_app.priority_for_current_frame(df, 51.4212, -0.1292, BALANCED)
        self.assertEqual(spy.call_count, 2)


class TestTheVersionCounterCannotBeBypassed(CacheTestCase):

    def test_the_setter_bumps_the_version(self):
        st_app.set_enriched_frame(a_frame())
        first = self.state['data_version']
        st_app.set_enriched_frame(a_frame())
        self.assertEqual(self.state['data_version'], first + 1)

    def test_the_setter_drops_the_stale_memo_outright(self):
        """Belt and braces: the version key alone would already force a miss,
        but holding the previous frame alive costs memory for nothing."""
        self.state['priority_cache'] = ('some key', a_frame())
        st_app.set_enriched_frame(a_frame())
        self.assertNotIn('priority_cache', self.state)

    def test_nothing_assigns_df_enriched_outside_the_setter(self):
        """The memo is only safe while `set_enriched_frame` is the single way
        in. A direct assignment at a new call site would serve scores computed
        from the frame before it.

        Checked with `ast` rather than by line number so that moving the
        function does not turn this into a test of where the file's lines
        happen to fall.
        """
        import ast

        with open('app/ui/st_app.py', 'r') as f:
            tree = ast.parse(f.read())

        setter = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == 'set_enriched_frame')
        allowed = {id(n) for n in ast.walk(setter)}

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute) and target.attr == 'df_enriched'
                        and id(node) not in allowed):
                    offenders.append(node.lineno)
        self.assertEqual(offenders, [],
                         f"st_app.py:{offenders} assigns df_enriched outside the setter")


if __name__ == '__main__':
    unittest.main()
