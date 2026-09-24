"""The table whose duplicate keys broke prediction (D-33).

`uk_postcode_demographics` is keyed on a postcode that every consumer
normalises with `REPLACE(UPPER(...), ' ', '')`, but this script chose what to
fetch with `SELECT DISTINCT PostCode` -- distinct *raw* spellings. `SW3 5UH`
and `SW3 5 UH` are two of those and one of this, so both were absent from the
target, both were fetched, and both were inserted.

Fifteen normalised postcodes ended up duplicated, one of them three times. What
that cost is in D-32: the demographics join fanned 103 restaurants into 226
rows, and `ML.PREDICT`'s MERGE refuses a source that matches a target twice --
after the Gemini pre-flight has been paid for.

The join is deduplicated now, so this is the second of the two fixes rather
than the load-bearing one. It is still worth making: the table should not
accumulate rows that every reader has to defend against.
"""
from unittest.mock import MagicMock, patch

from scripts.enrich_postcode_demographics import (
    build_missing_postcodes_query,
    enrich_postcodes,
)

MASTER = "p.d.fsa_master"
TARGET = "p.d.uk_postcode_demographics"


class TestItFetchesOneRowPerNormalisedPostcode:

    def test_it_groups_by_the_key_its_readers_join_on(self):
        sql = build_missing_postcodes_query(MASTER, TARGET)

        assert "GROUP BY REPLACE(UPPER(PostCode), ' ', '')" in sql

    def test_it_does_not_take_distinct_raw_spellings(self):
        """The whole defect in one line: two spellings, one key."""
        sql = build_missing_postcodes_query(MASTER, TARGET)

        assert 'DISTINCT PostCode' not in sql

    def test_it_prefers_the_spelling_without_the_stray_space(self):
        """`SW3 5 UH` is what postcodes.io answered with an all-NULL row, and
        it sorts *before* `SW3 5UH`, so a plain MIN picks the broken one. The
        shortest spelling is the one with no spaces where none belong."""
        sql = build_missing_postcodes_query(MASTER, TARGET)

        assert 'ORDER BY LENGTH(PostCode), PostCode' in sql

    def test_it_still_skips_postcodes_already_in_the_table(self):
        sql = build_missing_postcodes_query(MASTER, TARGET)

        assert f'FROM `{TARGET}`' in sql
        assert 'NOT IN' in sql

    def test_the_membership_test_uses_the_normalised_key_on_both_sides(self):
        """Comparing a raw spelling against a normalised one re-fetches every
        postcode whose stored spelling differs, forever."""
        sql = build_missing_postcodes_query(MASTER, TARGET)

        assert sql.count("REPLACE(UPPER(PostCode), ' ', '')") >= 2
        assert "REPLACE(UPPER(postcode), ' ', '')" in sql

    def test_the_limit_is_optional_and_interpolated_as_an_integer(self):
        """`ARRAY_AGG(... LIMIT 1)` has its own, so match the statement-level
        one by the GROUP BY it has to follow."""
        unlimited = build_missing_postcodes_query(MASTER, TARGET)
        limited = build_missing_postcodes_query(MASTER, TARGET, limit=10)

        assert not unlimited.rstrip().endswith('LIMIT 10')
        assert limited.rstrip().endswith('LIMIT 10')
        assert 'LIMIT 10' in limited.split('GROUP BY')[-1]


class TestTheFetchLoop:

    @patch('scripts.enrich_postcode_demographics.requests.post')
    @patch('scripts.enrich_postcode_demographics.ensure_demographics_table')
    @patch('scripts.enrich_postcode_demographics.bigquery.Client')
    def test_it_inserts_what_the_api_returned(self, mock_bq, _mock_ensure, mock_post):
        client = MagicMock()
        client.query.return_value.result.return_value = [MagicMock(postcode='SW16 1AA')]
        mock_bq.return_value = client
        client.insert_rows_json.return_value = []
        mock_post.return_value = MagicMock(status_code=200, **{'json.return_value': {
            'result': [{'query': 'SW16 1AA', 'result': {
                'lsoa': 'Lambeth 001A', 'msoa': 'Lambeth 001',
                'index_of_multiple_deprivation': 4242, 'admin_district': 'Lambeth'}}]}})

        written = enrich_postcodes('p', 'd')

        assert written == 1
        rows = client.insert_rows_json.call_args.args[1]
        assert rows == [{'postcode': 'SW16 1AA', 'lsoa': 'Lambeth 001A',
                         'msoa': 'Lambeth 001', 'imd_rank': 4242,
                         'admin_district': 'Lambeth'}]

    @patch('scripts.enrich_postcode_demographics.requests.post')
    @patch('scripts.enrich_postcode_demographics.ensure_demographics_table')
    @patch('scripts.enrich_postcode_demographics.bigquery.Client')
    def test_nothing_missing_is_not_an_error(self, mock_bq, _mock_ensure, mock_post):
        client = MagicMock()
        client.query.return_value.result.return_value = []
        mock_bq.return_value = client

        assert enrich_postcodes('p', 'd') == 0
        mock_post.assert_not_called()
