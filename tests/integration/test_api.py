import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Adding proejct root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------
# 1. ROOT ENDPOINT TEST
# ---------------------------------------------------------
def test_root_endpoint():
    """Checks if the API is up and running."""
    response = client.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------
# 2. PREDICT ENDPOINT - CACHE HIT (REDIS POPULATED)
# ---------------------------------------------------------
@patch("src.api.main.redis_available", True)
@patch("src.api.main.model")
@patch("src.api.main.cache")
def test_predict_cache_hit(mock_cache, mock_model):
    """
    Tests that the model is NEVER triggered if data exists in Redis (Cache Hit).
    """
    # Standard synchronous return value
    mock_cache.get.return_value = '{"predicted_duration_seconds": 850.5, "predicted_duration_minutes": 14.17}'

    payload = {
        "pickup_datetime": "2026-01-20 12:00:00",
        "dropoff_datetime": "2026-01-20 12:15:00",
        "passenger_count": 1,
        "pickup_longitude": -73.9857,
        "pickup_latitude": 40.7484,
        "dropoff_longitude": -73.9665,
        "dropoff_latitude": 40.7812,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_duration_seconds"] == 850.5

    # Model should not be called on cache hit
    mock_model.run.assert_not_called()


# ---------------------------------------------------------
# 3. PREDICT ENDPOINT - CACHE MISS (REDIS EMPTY)
# ---------------------------------------------------------
@patch("src.api.main.redis_available", True)
@patch("src.api.main.model")
@patch("src.api.main.cache")
def test_predict_cache_miss(mock_cache, mock_model):
    """
    Tests that data goes to the model if Redis is empty (Cache Miss).
    """
    mock_cache.get.return_value = None
    mock_cache.setex.return_value = True

    # Simulate model prediction
    mock_output = np.array([[2.7]])
    mock_model.run.return_value = [mock_output]

    payload = {
        "pickup_datetime": "2026-01-20 12:00:00",
        "dropoff_datetime": "2026-01-20 12:15:00",
        "passenger_count": 1,
        "pickup_longitude": -73.9857,
        "pickup_latitude": 40.7484,
        "dropoff_longitude": -73.9665,
        "dropoff_latitude": 40.7812,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_duration_seconds" in data
    
    mock_model.run.assert_called_once()
    mock_cache.setex.assert_called_once()


# ---------------------------------------------------------
# 4. PREDICT ENDPOINT - INVALID DATA (EDGE CASES)
# ---------------------------------------------------------
@pytest.mark.parametrize(
    "invalid_payload, expected_status",
    [
        ({"pickup_datetime": "2026-01-20 12:00:00"}, 422),
        (
            {
                "pickup_datetime": "2026-01-20 12:00:00",
                "dropoff_datetime": "2026-01-20 12:15:00",
                "passenger_count": "invalid_type",
                "pickup_longitude": -73.9857,
                "pickup_latitude": 40.7484,
                "dropoff_longitude": -73.9665,
                "dropoff_latitude": 40.7812,
            },
            422,
        ),
        (
            {
                "pickup_datetime": "2026-01-20 12:00:00",
                "dropoff_datetime": "2026-01-20 12:15:00",
                "passenger_count": 1,
                "pickup_longitude": -73.9857,
                "pickup_latitude": 150.0000,
                "dropoff_longitude": -73.9665,
                "dropoff_latitude": 40.7812,
            },
            422,
        ),
    ],
)
def test_predict_invalid_data(invalid_payload, expected_status):
    """Tests API behavior with invalid input data."""
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == expected_status