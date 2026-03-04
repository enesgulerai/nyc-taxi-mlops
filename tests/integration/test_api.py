import os
import sys
from unittest.mock import patch, AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Note: In the future, we will resolve this sys.path issue with pytest.ini or pyproject.toml
# but keeping it for now to ensure the code runs.
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
    # Optional: You can also check the returned JSON message
    # assert response.json() == {"status": "ok", "message": "API is running"}


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
    # Since decode_responses=True, we return a string.
    # Added both seconds and minutes to match your schema.
    mock_cache.get = AsyncMock(
        return_value='{"predicted_duration_seconds": 850.5, "predicted_duration_minutes": 14.17}'
    )

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

    # If data came from cache, the model should NEVER run!
    mock_model.run.assert_not_called()


# ---------------------------------------------------------
# 3. PREDICT ENDPOINT - CACHE MISS (REDIS EMPTY)
# ---------------------------------------------------------
@patch("src.api.main.redis_available", True)
@patch("src.api.main.model")
@patch("src.api.main.cache")
def test_predict_cache_miss(mock_cache, mock_model):
    """
    Tests that the data goes to the model and generates a prediction if Redis is empty (Cache Miss).
    """
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.setex = AsyncMock(return_value=True)

    # Simulate model prediction (logarithmic value)
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

    assert response.status_code == 200, f"API Error: {response.text}"

    data = response.json()
    assert "predicted_duration_seconds" in data
    assert data["predicted_duration_seconds"] > 0

    mock_model.run.assert_called_once()
    mock_cache.setex.assert_called_once()  # We also tested that it is written to the cache!


# ---------------------------------------------------------
# 4. PREDICT ENDPOINT - INVALID DATA (EDGE CASES)
# ---------------------------------------------------------
@pytest.mark.parametrize(
    "invalid_payload, expected_status",
    [
        # Scenario 1: Missing data (Only datetime is present)
        ({"pickup_datetime": "2026-01-20 12:00:00"}, 422),
        # Scenario 2: Wrong Data Type (passenger_count must be integer, string provided)
        (
            {
                "pickup_datetime": "2026-01-20 12:00:00",
                "dropoff_datetime": "2026-01-20 12:15:00",
                "passenger_count": "one_passenger",
                "pickup_longitude": -73.9857,
                "pickup_latitude": 40.7484,
                "dropoff_longitude": -73.9665,
                "dropoff_latitude": 40.7812,
            },
            422,
        ),
        # Scenario 3: Illogical Coordinate (If Pydantic schema has validation, this should return 422)
        # Latitude must be between -90 and 90.
        (
            {
                "pickup_datetime": "2026-01-20 12:00:00",
                "dropoff_datetime": "2026-01-20 12:15:00",
                "passenger_count": 1,
                "pickup_longitude": -73.9857,
                "pickup_latitude": 150.0000,  # Invalid latitude
                "dropoff_longitude": -73.9665,
                "dropoff_latitude": 40.7812,
            },
            422,
        ),
    ],
)
def test_predict_invalid_data(invalid_payload, expected_status):
    """Tests that the API does not crash and returns the correct error code under various invalid data scenarios."""
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == expected_status