"""What a prediction run pays for, and what it declines to pay for again.

Every collaborator is mocked; these tests spend nothing. The single question
they ask is which of Places, Gemini and postcodes.io `generate_predictions`
decides to call for a given row.

`FindRow` mirrors the projection of the target-batch query in
`ml_prediction.py`. It fell out of sync once already -- the double still
carried `maps_rating` and the raw profile JSON long after Phases 5 and 6 moved
both guards onto timestamps, so all three tests here errored with
`'DummyRow' object has no attribute 'maps_lookup_at'`. Nothing noticed, because
`tests/` was never collected by CI. That is D11, and this file is one of the
things that was rotting behind it.
"""
import contextlib
import datetime
import types
from unittest.mock import MagicMock, patch

from app.services.ml_prediction import generate_predictions

_UTC = datetime.timezone.utc

# Both backfilled to the Phase 5 migration date, which is what production rows
# carry. `STALE` is comfortably past GEMINI_PROFILE_MAX_AGE_DAYS (180).
LOOKED_UP = datetime.datetime(2026, 9, 23, tzinfo=_UTC)
PROFILED = datetime.datetime(2026, 9, 23, tzinfo=_UTC)
STALE = datetime.datetime(2025, 1, 1, tzinfo=_UTC)


class FindRow:
    """One row of the target-batch query.

    Keyword-only and fully defaulted to "nothing is missing", so each test
    states only the one field it is about. The defaults are the expensive
    direction to get wrong: a row that looks complete cannot make a test pass
    by accidentally triggering enrichment.
    """

    def __init__(self, fhrsid, *, postcode='SW16 1AA', maps_lookup_at=LOOKED_UP,
                 gemini_profiled_at=PROFILED, has_profile=True,
                 d_postcode='SW16 1AA'):
        self.fhrsid = fhrsid
        self.postcode = postcode
        self.maps_lookup_at = maps_lookup_at
        self.gemini_profiled_at = gemini_profiled_at
        self.has_profile = has_profile
        self.d_postcode = d_postcode


@contextlib.contextmanager
def predicting(rows):
    """Run `generate_predictions` against `rows` with every spend patched out.

    `enrich_postcodes` is imported inside the function body, so it is patched
    where it is defined rather than where it is used -- unpatched, this file
    would make a real postcodes.io call from the offline suite.
    """
    with patch('app.services.ml_prediction.bigquery.Client') as client_cls, \
         patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid') as maps, \
         patch('app.services.ml_prediction.execute_gemini_enrichment') as gemini, \
         patch('scripts.enrich_postcode_demographics.enrich_postcodes') as postcodes:
        find_job = MagicMock()
        find_job.result.return_value = rows
        predict_job = MagicMock()
        predict_job.num_dml_affected_rows = len(rows)
        client_cls.return_value.query.side_effect = [find_job, predict_job]
        yield types.SimpleNamespace(maps=maps, gemini=gemini, postcodes=postcodes)


def test_a_complete_row_triggers_no_enrichment_at_all():
    with predicting([FindRow('123')]) as calls:
        success, _ = generate_predictions(
            'project', 'dataset', 'table', 'model', target_fhrsids=['123'])

    assert success is True
    calls.maps.assert_not_called()
    calls.gemini.assert_not_called()
    calls.postcodes.assert_not_called()


def test_a_row_never_looked_up_goes_to_places():
    with predicting([FindRow('124', maps_lookup_at=None)]) as calls:
        success, _ = generate_predictions(
            'project', 'dataset', 'table', 'model', target_fhrsids=['124'])

    assert success is True
    calls.maps.assert_called_once_with(['124'], limit=1, force_regen=False)
    calls.gemini.assert_not_called()


def test_a_permanent_places_miss_is_not_looked_up_again():
    """The Phase 5 point, and the one the old double could not express: a
    stamped lookup with nothing to show for it is answered, not pending."""
    with predicting([FindRow('125', maps_lookup_at=LOOKED_UP)]) as calls:
        generate_predictions(
            'project', 'dataset', 'table', 'model', target_fhrsids=['125'])

    calls.maps.assert_not_called()


def test_an_unprofiled_row_goes_to_gemini():
    with predicting([FindRow('126', has_profile=False,
                             gemini_profiled_at=None)]) as calls:
        success, _ = generate_predictions(
            'project', 'dataset', 'table', 'model', target_fhrsids=['126'])

    assert success is True
    calls.gemini.assert_called_once_with(
        'project', 'dataset', 'table', fhrsids=['126'])
    calls.maps.assert_not_called()


def test_a_profile_older_than_the_max_age_is_refreshed():
    with predicting([FindRow('127', gemini_profiled_at=STALE)]) as calls:
        generate_predictions(
            'project', 'dataset', 'table', 'model', target_fhrsids=['127'])

    calls.gemini.assert_called_once_with(
        'project', 'dataset', 'table', fhrsids=['127'])


def test_a_postcode_absent_from_the_demographics_table_is_backfilled():
    with predicting([FindRow('128', d_postcode=None)]) as calls:
        generate_predictions(
            'project', 'dataset', 'table', 'model', target_fhrsids=['128'])

    calls.postcodes.assert_called_once_with(
        project_id='project', dataset_id='dataset', master_table='table')


def test_the_force_flags_regenerate_data_that_is_already_there():
    with predicting([FindRow('129')]) as calls:
        success, _ = generate_predictions(
            'project', 'dataset', 'table', 'model', target_fhrsids=['129'],
            force_maps=True, force_gemini=True)

    assert success is True
    calls.maps.assert_called_once_with(['129'], limit=1, force_regen=True)
    calls.gemini.assert_called_once_with(
        'project', 'dataset', 'table', fhrsids=['129'])
