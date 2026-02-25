import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Not: Gelecekte bu sys.path olayını pytest.ini veya pyproject.toml ile çözeceğiz
# ama şimdilik kodun çalışması için tutuyorum.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------
# 1. ROOT ENDPOINT TEST
# ---------------------------------------------------------
def test_root_endpoint():
    """API'nin ayakta olup olmadığını kontrol eder."""
    response = client.get("/")
    assert response.status_code == 200
    # Opsiyonel: Dönen JSON mesajını da kontrol edebilirsin
    # assert response.json() == {"status": "ok", "message": "API is running"}

# ---------------------------------------------------------
# 2. PREDICT ENDPOINT - CACHE HIT (REDIS DOLU)
# ---------------------------------------------------------
@patch("src.api.main.redis_available", True)  # <-- YENİ EKLENDİ
@patch("src.api.main.model")
@patch("src.api.main.cache")
def test_predict_cache_hit(mock_cache, mock_model):
    """
    Redis'te veri varsa (Cache Hit), modelin ASLA tetiklenmemesi gerektiğini test eder.
    """
    # decode_responses=True olduğu için string dönüyoruz.
    # Şemana uygun olarak hem seconds hem minutes ekledik.
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
    
    # Veri cache'ten geldiyse model ASLA çalışmamalı!
    mock_model.run.assert_not_called()

# ---------------------------------------------------------
# 3. PREDICT ENDPOINT - CACHE MISS (REDIS BOŞ)
# ---------------------------------------------------------
@patch("src.api.main.redis_available", True)  # <-- YENİ EKLENDİ
@patch("src.api.main.model")
@patch("src.api.main.cache")
def test_predict_cache_miss(mock_cache, mock_model):
    """
    Redis boşsa (Cache Miss), verinin modele gidip tahmin üretildiğini test eder.
    """
    mock_cache.get.return_value = None
    mock_cache.setex.return_value = True  # main.py'da setex kullanmışsın

    # Model tahmini taklit et (logaritmik değer)
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
    mock_cache.setex.assert_called_once() # Cache'e yazıldığını da test etmiş olduk!

# ---------------------------------------------------------
# 4. PREDICT ENDPOINT - INVALID DATA (EDGE CASES)
# ---------------------------------------------------------
@pytest.mark.parametrize(
    "invalid_payload, expected_status",
    [
        # Senaryo 1: Eksik veri (Sadece datetime var)
        ({"pickup_datetime": "2026-01-20 12:00:00"}, 422),
        
        # Senaryo 2: Yanlış Veri Tipi (passenger_count integer olmalı, string verilmiş)
        ({
            "pickup_datetime": "2026-01-20 12:00:00",
            "dropoff_datetime": "2026-01-20 12:15:00",
            "passenger_count": "bir_yolcu", 
            "pickup_longitude": -73.9857,
            "pickup_latitude": 40.7484,
            "dropoff_longitude": -73.9665,
            "dropoff_latitude": 40.7812,
        }, 422),

        # Senaryo 3: Mantıksız Koordinat (Pydantic şemanda validation varsa bu 422 dönmeli)
        # Enlem (latitude) -90 ile 90 arasında olmalıdır.
        ({
            "pickup_datetime": "2026-01-20 12:00:00",
            "dropoff_datetime": "2026-01-20 12:15:00",
            "passenger_count": 1,
            "pickup_longitude": -73.9857,
            "pickup_latitude": 150.0000, # Hatalı enlem
            "dropoff_longitude": -73.9665,
            "dropoff_latitude": 40.7812,
        }, 422),
    ]
)
def test_predict_invalid_data(invalid_payload, expected_status):
    """Farklı hatalı veri senaryolarında API'nin çökmediğini ve doğru hata kodunu döndüğünü test eder."""
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == expected_status