"""The refill half of `invalidate_stale_predictions.py`.

D-20 cleared 1,065 predictions written by a model that had been replaced. That
was right, and nothing put any back: `predicted_user_rating` is NULL on all
11,268 rows, so the priority heuristic's staleness component scores a constant
100 and stops discriminating between rows entirely.

The rows this script scores are the ones that need nothing bought for them --
profile present and fresh, Maps looked up, demographics joined -- so the whole
refill costs `ML.PREDICT` and nothing else. The load-bearing word is *needs*:
if the guarantee is wrong the script bills Places and `AI.GENERATE` for
thousands of rows, so it re-derives the answer at run time through the same
function production uses, and refuses rather than spending.
"""
from unittest.mock import MagicMock, patch

import pytest

from scripts.backfill_predictions import (
    DEFAULT_BQ_PATH,
    build_candidate_query,
    run_backfill,
)


class Candidate:
    def __init__(self, fhrsid):
        self.fhrsid = fhrsid


class FindRow:
    """A row as the production find query returns it."""

    def __init__(self, fhrsid, maps_lookup_at='2026-01-01 00:00:00+00:00',
                 has_profile=True, gemini_profiled_at='2026-09-01 00:00:00+00:00',
                 postcode='SW16 1AA', d_postcode='SW16 1AA'):
        self.fhrsid = fhrsid
        self.maps_lookup_at = maps_lookup_at
        self.has_profile = has_profile
        self.gemini_profiled_at = gemini_profiled_at
        self.postcode = postcode
        self.d_postcode = d_postcode


def _mock_client(candidates, find_rows=None):
    """A client that answers the candidate query and the verification query.

    Dispatches on the SQL rather than on call order, so a test does not silently
    depend on how many times the script looks something up.
    """
    client = MagicMock()

    def query(sql, **kwargs):
        job = MagicMock()
        if 'AS has_profile' in sql:
            ids = {c.fhrsid for c in candidates}
            rows = find_rows if find_rows is not None else [FindRow(i) for i in sorted(ids)]
            job.result.return_value = [r for r in rows if str(r.fhrsid) in
                                       {i.strip("'") for i in sql.split('IN (')[-1].split(')')[0].split(', ')}]
        else:
            job.result.return_value = candidates
        job.total_bytes_processed = 1024 * 1024
        return job

    client.query.side_effect = query
    return client


class TestTheCandidateQueryPicksOnlyFreeRows:

    def test_it_asks_for_rows_with_no_prediction(self):
        sql = build_candidate_query(DEFAULT_BQ_PATH)

        assert 'predicted_user_rating IS NULL' in sql

    def test_it_requires_a_profile_and_a_maps_lookup(self):
        sql = build_candidate_query(DEFAULT_BQ_PATH)

        assert 'gemini_insights_structured IS NOT NULL' in sql
        assert 'maps_lookup_at IS NOT NULL' in sql

    def test_it_excludes_profiles_stale_by_the_shared_threshold(self):
        """A stale profile is one `generate_predictions` would re-buy. Hard-coding
        the number here is how the two drift apart."""
        from app.core.profile_freshness import GEMINI_PROFILE_MAX_AGE_DAYS

        sql = build_candidate_query(DEFAULT_BQ_PATH)

        assert f'INTERVAL {GEMINI_PROFILE_MAX_AGE_DAYS} DAY' in sql

    def test_a_row_whose_postcode_does_not_join_is_not_a_candidate(self):
        sql = build_candidate_query(DEFAULT_BQ_PATH)

        assert 'uk_postcode_demographics' in sql
        assert 'EXISTS' in sql, "a LEFT JOIN here can fan out on duplicate postcodes"

    def test_a_row_with_no_postcode_at_all_is_still_a_candidate(self):
        """Nothing can be looked up for it, so nothing will be bought for it."""
        sql = build_candidate_query(DEFAULT_BQ_PATH)

        assert 'postcode IS NULL' in sql


class TestTheDryRunIsTheDefault:

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_it_scores_nothing(self, mock_bq, mock_predict):
        mock_bq.return_value = _mock_client([Candidate('1'), Candidate('2')])

        result = run_backfill(DEFAULT_BQ_PATH)

        mock_predict.assert_not_called()
        assert result['candidates'] == 2

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_it_still_verifies_every_chunk(self, mock_bq, mock_predict):
        """The point of the dry run is to prove the batch is free *before*
        anyone types --execute, so the verification has to happen without it."""
        mock_bq.return_value = _mock_client([Candidate(str(i)) for i in range(5)])

        result = run_backfill(DEFAULT_BQ_PATH, chunk_size=2)

        assert result['chunks'] == 3
        assert result['verified'] == 5
        mock_predict.assert_not_called()


class TestItRefusesRatherThanSpends:

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_an_unprofiled_row_aborts_the_run(self, mock_bq, mock_predict):
        mock_bq.return_value = _mock_client(
            [Candidate('1')], find_rows=[FindRow('1', has_profile=False,
                                                 gemini_profiled_at=None)])

        with pytest.raises(RuntimeError, match='gemini'):
            run_backfill(DEFAULT_BQ_PATH, execute=True)

        mock_predict.assert_not_called()

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_a_row_never_looked_up_in_maps_aborts_the_run(self, mock_bq, mock_predict):
        mock_bq.return_value = _mock_client(
            [Candidate('1')], find_rows=[FindRow('1', maps_lookup_at=None)])

        with pytest.raises(RuntimeError, match='maps'):
            run_backfill(DEFAULT_BQ_PATH, execute=True)

        mock_predict.assert_not_called()

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_an_unjoined_postcode_aborts_the_run(self, mock_bq, mock_predict):
        mock_bq.return_value = _mock_client(
            [Candidate('1')], find_rows=[FindRow('1', d_postcode=None)])

        with pytest.raises(RuntimeError, match='postcodes'):
            run_backfill(DEFAULT_BQ_PATH, execute=True)

        mock_predict.assert_not_called()

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_the_dry_run_refuses_too(self, mock_bq, mock_predict):
        """Otherwise the dry run reports a clean batch and --execute discovers
        the bill, which is the wrong order for that discovery."""
        mock_bq.return_value = _mock_client(
            [Candidate('1')], find_rows=[FindRow('1', maps_lookup_at=None)])

        with pytest.raises(RuntimeError):
            run_backfill(DEFAULT_BQ_PATH)

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_an_earlier_chunk_is_not_scored_before_a_later_one_is_checked(
            self, mock_bq, mock_predict):
        """Chunk 1 is clean, chunk 2 is not. Verifying inside the scoring loop
        would bill chunk 1 and then abort; the whole batch is checked first."""
        mock_bq.return_value = _mock_client(
            [Candidate('1'), Candidate('2')],
            find_rows=[FindRow('1'), FindRow('2', maps_lookup_at=None)])

        with pytest.raises(RuntimeError):
            run_backfill(DEFAULT_BQ_PATH, execute=True, chunk_size=1)

        mock_predict.assert_not_called()


class TestExecuteScoresInChunks:

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_it_scores_every_candidate_exactly_once(self, mock_bq, mock_predict):
        mock_predict.return_value = (True, 'ok')
        mock_bq.return_value = _mock_client([Candidate(str(i)) for i in range(5)])

        result = run_backfill(DEFAULT_BQ_PATH, execute=True, chunk_size=2)

        assert mock_predict.call_count == 3
        scored = [fid for call in mock_predict.call_args_list
                  for fid in call.kwargs['target_fhrsids']]
        assert sorted(scored) == ['0', '1', '2', '3', '4']
        assert result['scored'] == 5

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_it_never_forces_an_enrichment(self, mock_bq, mock_predict):
        """`force_maps` or `force_gemini` would turn the free run into the
        expensive one, which is the exact failure this script exists to avoid."""
        mock_predict.return_value = (True, 'ok')
        mock_bq.return_value = _mock_client([Candidate('1')])

        run_backfill(DEFAULT_BQ_PATH, execute=True)

        assert mock_predict.call_args.kwargs.get('force_maps', False) is False
        assert mock_predict.call_args.kwargs.get('force_gemini', False) is False

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_a_failed_chunk_stops_the_run(self, mock_bq, mock_predict):
        mock_predict.return_value = (False, 'ML.PREDICT failed: no such model')
        mock_bq.return_value = _mock_client([Candidate(str(i)) for i in range(4)])

        with pytest.raises(RuntimeError, match='no such model'):
            run_backfill(DEFAULT_BQ_PATH, execute=True, chunk_size=2)

        assert mock_predict.call_count == 1

    @patch('scripts.backfill_predictions.generate_predictions')
    @patch('scripts.backfill_predictions.bigquery.Client')
    def test_nothing_to_do_is_not_an_error(self, mock_bq, mock_predict):
        mock_bq.return_value = _mock_client([])

        result = run_backfill(DEFAULT_BQ_PATH, execute=True)

        assert result['candidates'] == 0
        mock_predict.assert_not_called()
