import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as rt
import pandas as pd
import redis.asyncio as redis  # 1. DEĞİŞİKLİK: Asenkron Redis
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import PredictionOutput, TaxiInput
from src.components.feature_engineering import create_features
from src.config import MODEL_SAVE_PATH
from src.utils.logger import get_logger

# LOGGER
logger = get_logger("api_service")

# GLOBAL VARIABLES
model = None
input_name = None
cache = None
redis_available = False


# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, input_name, cache, redis_available

    # 1. REDIS (ASYNC)
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    try:
        cache = redis.Redis(
            host=REDIS_HOST, port=6379, decode_responses=True, socket_connect_timeout=1
        )
        await cache.ping()  # Await eklendi
        redis_available = True
        logger.info(f"✅ REDIS CONNECTED: {REDIS_HOST}")
    except Exception as e:
        logger.warning(f"⚠️ REDIS FAILED: {e}")
        redis_available = False

    # 2. LOAD THE MODEL
    try:
        model = rt.InferenceSession(MODEL_SAVE_PATH)
        input_name = model.get_inputs()[0].name
        logger.info(f"✅ MODEL LOADED: {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"❌ MODEL LOAD ERROR: {e}")
        raise e

    yield

    # 3. CLEANUP
    if cache:
        await cache.close()  # Await eklendi
    logger.info("🛑 SHUTDOWN")


# --- APP INITIALIZATION ---
app = FastAPI(title="NYC Taxi API", version="2.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


def generate_cache_key(data: TaxiInput) -> str:
    data_str = json.dumps(data.model_dump(), sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


# 2. DEĞİŞİKLİK: CPU-Bound işlemleri izole eden yeni fonksiyon
def run_inference(data_dict: dict) -> float:
    """
    Pandas ve ONNX gibi saf işlemci gücü isteyen ve sistemi kilitleyen
    işlemleri Event Loop'tan ayırmak için arka planda çalıştırılır.
    """
    df = pd.DataFrame([data_dict])
    df = create_features(df)

    features = [
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "month",
        "day_of_week",
        "hour",
        "is_weekend",
        "distance_haversine",
        "distance_manhattan",
        "bearing",
    ]

    X = df[features].astype(np.float32).to_numpy()

    # Inference
    results = model.run(None, {input_name: X})

    log_pred = results[0].item()
    return float(np.expm1(log_pred))


@app.get("/")
def root():
    return {"message": "NYC TAXI PREDICTION API IS LIVE"}


# 3. DEĞİŞİKLİK: Async Endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: TaxiInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model service not ready")

    try:
        # 1. CACHE CHECK (ASYNC)
        cache_key = generate_cache_key(data)
        if redis_available:
            cached = await cache.get(cache_key)  # Await eklendi
            if cached:
                logger.info("⚡ CACHE HIT")
                return json.loads(cached)

        # 2. PREDICTION (THREAD POOL'A GÖNDERİLİYOR)
        # API ana damarını tıkamamak için ağır işi arka plandaki işçilere veriyoruz.
        pred_seconds = await asyncio.to_thread(run_inference, data.model_dump())

        response = {
            "predicted_duration_seconds": round(pred_seconds, 2),
            "predicted_duration_minutes": round(pred_seconds / 60, 2),
        }

        # 3. CACHE SAVE (ASYNC)
        if redis_available:
            await cache.setex(cache_key, 3600, json.dumps(response))  # Await eklendi

        return response

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
