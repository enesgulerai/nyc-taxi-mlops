import os
import pathlib

import mlflow
import mlflow.sklearn
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import load_and_clean_data
from src.components.feature_engineering import create_features

# Project Modules
from src.config import DATA_RAW_PATH, MLFLOW_EXPERIMENT_NAME, MODEL_SAVE_PATH
from src.utils.logger import get_logger

logger = get_logger("training_pipeline")


def run_training():
    """
    Executes the training pipeline with FIXED PRODUCTION PARAMETERS.
    No loop, single run, optimized for deployment.
    """
    try:
        logger.info("🚀 PRODUCTION TRAINING PIPELINE INITIALIZED")

        # ---------------------------------------------------------
        # 1. MLFLOW CONNECTION SETUP
        # ---------------------------------------------------------
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if not tracking_uri:
            mlruns_path = pathlib.Path("./mlruns").resolve()
            tracking_uri = mlruns_path.as_uri()
            logger.warning(
                f"⚠️ No MLFLOW_TRACKING_URI found. Using Local File Store: {tracking_uri}"
            )
        else:
            logger.info(f"📡 Connecting to MLflow Server at: {tracking_uri}")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # ---------------------------------------------------------
        # 2. DATA LOADING & PREPROCESSING
        # ---------------------------------------------------------
        logger.info("💾 LOADING AND CLEANING RAW DATA...")

        if not os.path.exists(DATA_RAW_PATH):
            abs_data_path = os.path.abspath(DATA_RAW_PATH)
            raise FileNotFoundError(f"❌ DATA FILE NOT FOUND AT: {abs_data_path}")

        df = load_and_clean_data(DATA_RAW_PATH)

        logger.info("🛠️ APPLYING FEATURE ENGINEERING...")
        df_processed = create_features(df)

        # Velocity Filter
        df_processed["avg_speed_kph"] = (
            df_processed["distance_haversine"] / df_processed["trip_duration"]
        ) * 3600
        df_processed = df_processed[
            (df_processed["avg_speed_kph"] <= 100)
            & (df_processed["avg_speed_kph"] >= 0.1)
        ]

        df_processed["trip_duration_log"] = np.log1p(df_processed["trip_duration"])

        # Features
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
        target = "trip_duration_log"

        X = df_processed[features]
        y = df_processed[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------------------------------------------------
        # 3. PRODUCTION TRAINING (BEST PARAMS) 🏆
        # ---------------------------------------------------------

        prod_params = {
            "n_estimators": 122,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "random_state": 42,
            "n_jobs": -1,
        }

        logger.info(f"🏭 STARTING TRAINING WITH PRODUCTION PARAMS: {prod_params}")

        with mlflow.start_run(run_name="Production_Best_Model"):
            mlflow.log_params(prod_params)

            model = RandomForestRegressor(**prod_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            logger.info(f"✅ MODEL TRAINED | RMSE: {rmse:.4f}")

            # ---------------------------------------------------------
            # 4. EXPORT TO ONNX
            # ---------------------------------------------------------
            logger.info("📦 EXPORTING ONNX MODEL...")

            initial_type = [("float_input", FloatTensorType([None, len(features)]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

            save_dir = os.path.dirname(MODEL_SAVE_PATH)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(MODEL_SAVE_PATH, "wb") as f:
                f.write(onnx_model.SerializeToString())

            # Artifact Loglama
            mlflow.log_artifact(MODEL_SAVE_PATH, artifact_path="onnx_model")

            size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
            logger.info(f"📉 FINAL MODEL SIZE: {size_mb:.2f} MB")

            if size_mb > 100:
                logger.warning(
                    "⚠️ MODEL SIZE IS LARGE! This might cause OOM errors in Kubernetes."
                )

        logger.info("🏁 TRAINING PIPELINE FINISHED")

    except Exception as e:
        logger.error(f"❌ FAILURE: {e}")
        raise e


if __name__ == "__main__":
    run_training()
