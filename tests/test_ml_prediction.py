import pytest
from unittest.mock import patch, MagicMock
from app.services.ml_prediction import generate_predictions

class DummyRow:
    def __init__(self, fhrsid, maps_rating, gemini_insights):
        self.fhrsid = fhrsid
        self.maps_rating = maps_rating
        self.gemini_insights = gemini_insights

@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_generate_predictions_skips_when_data_exists(mock_execute_gemini, mock_enrich_maps, mock_bq_client):
    mock_client_instance = MagicMock()
    mock_bq_client.return_value = mock_client_instance
    
    mock_query_job = MagicMock()
    # Mocking that the targeted restaurant already has both maps_rating and gemini_insights
    mock_query_job.result.return_value = [
        DummyRow('123', 4.5, '{"match_score": 90}')
    ]
    
    # Second query for prediction
    mock_predict_job = MagicMock()
    mock_predict_job.num_dml_affected_rows = 1
    
    mock_client_instance.query.side_effect = [mock_query_job, mock_predict_job]
    
    success, msg = generate_predictions(
        'project', 'dataset', 'table', 'model', 
        target_fhrsids=['123'], force_maps=False, force_gemini=False
    )
    
    assert success is True
    # Should skip auto-enrichment because maps and gemini are populated
    mock_enrich_maps.assert_not_called()
    mock_execute_gemini.assert_not_called()

@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_generate_predictions_triggers_when_data_missing(mock_execute_gemini, mock_enrich_maps, mock_bq_client):
    mock_client_instance = MagicMock()
    mock_bq_client.return_value = mock_client_instance
    
    mock_query_job = MagicMock()
    # Data is missing maps_rating and gemini_insights
    mock_query_job.result.return_value = [
        DummyRow('124', None, None)
    ]
    
    mock_predict_job = MagicMock()
    mock_predict_job.num_dml_affected_rows = 1
    
    mock_client_instance.query.side_effect = [mock_query_job, mock_predict_job]
    
    success, msg = generate_predictions(
        'project', 'dataset', 'table', 'model', 
        target_fhrsids=['124'], force_maps=False, force_gemini=False
    )
    
    assert success is True
    # Should trigger auto-enrichment because data is missing
    mock_enrich_maps.assert_called_once_with(['124'], limit=1, force_regen=False)
    mock_execute_gemini.assert_called_once_with('project', 'dataset', 'table', fhrsids=['124'])

@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_generate_predictions_forces_regeneration(mock_execute_gemini, mock_enrich_maps, mock_bq_client):
    mock_client_instance = MagicMock()
    mock_bq_client.return_value = mock_client_instance
    
    mock_query_job = MagicMock()
    # Data exists, but we'll force regen
    mock_query_job.result.return_value = [
        DummyRow('125', 4.5, '{"match_score": 90}')
    ]
    
    mock_predict_job = MagicMock()
    mock_predict_job.num_dml_affected_rows = 1
    
    mock_client_instance.query.side_effect = [mock_query_job, mock_predict_job]
    
    success, msg = generate_predictions(
        'project', 'dataset', 'table', 'model', 
        target_fhrsids=['125'], force_maps=True, force_gemini=True
    )
    
    assert success is True
    # Should trigger auto-enrichment despite data existing because of force flags
    mock_enrich_maps.assert_called_once_with(['125'], limit=1, force_regen=True)
    mock_execute_gemini.assert_called_once_with('project', 'dataset', 'table', fhrsids=['125'])
