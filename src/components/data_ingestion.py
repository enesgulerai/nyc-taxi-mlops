import os
import zipfile

import gdown
import pandas as pd

from src.config import NYC_BOUNDS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- GOOGLE DRIVE SETTINGS ---
DRIVE_FILE_ID = "1bC2VJsYYQdDOUKMdu4W6YPP0NPgQqBbc"


def check_and_download_data(filepath: str):
    """
    If the file doesn't exist at the given file path, it downloads it from Google Drive and extracts it from the zip file.
    It also automatically creates subfolders such as 'data/raw'.
    """
    if os.path.exists(filepath):
        logger.info(f"✅ DATA FOUND AT: {filepath}")
        return

    logger.info(
        f"⬇️ DATA NOT FOUND. DOWNLOADING FROM GOOGLE DRIVE (ID: {DRIVE_FILE_ID})..."
    )

    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"📁 Created directory: {directory}")

    zip_path = os.path.join(directory, "temp_data.zip")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

    try:
        # 1. DOWNLOAD
        gdown.download(url, zip_path, quiet=False)

        # 2. EXTRACTING
        if os.path.exists(zip_path):
            logger.info("📦 EXTRACTING ZIP FILE...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(directory)

            # 3. CLEANING
            os.remove(zip_path)
            logger.info(
                f"✅ DOWNLOAD & EXTRACTION COMPLETE. File is ready at: {filepath}"
            )
        else:
            raise FileNotFoundError("Downloaded zip file could not be found.")

    except Exception as e:
        logger.error(f"❌ DOWNLOAD FAILED: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise e


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    check_and_download_data(filepath)

    logger.info(f"LOADING DATA FROM: {filepath}")

    if not os.path.exists(filepath):
        logger.error(f"FILE {filepath} DOES NOT EXIST")
        raise FileNotFoundError(f"FILE COULD NOT FIND: {filepath}")

    df = pd.read_csv(filepath)
    original_len = len(df)

    # 1. CLEANING: TIME
    df = df[(df["trip_duration"] >= 60) & (df["trip_duration"] <= 10800)]

    # 2. CLEANING: COORDINATE BOUNDARIES
    df = df[
        (df["pickup_longitude"] >= NYC_BOUNDS["min_lng"])
        & (df["pickup_longitude"] <= NYC_BOUNDS["max_lng"])
        & (df["pickup_latitude"] >= NYC_BOUNDS["min_lat"])
        & (df["pickup_latitude"] <= NYC_BOUNDS["max_lat"])
        & (df["dropoff_longitude"] >= NYC_BOUNDS["min_lng"])
        & (df["dropoff_longitude"] <= NYC_BOUNDS["max_lng"])
        & (df["dropoff_latitude"] >= NYC_BOUNDS["min_lat"])
        & (df["dropoff_latitude"] <= NYC_BOUNDS["max_lat"])
    ]

    logger.info(
        f"THE CLEANUP IS COMPLETE. THE REMAINING LINES ARE {original_len} -> {len(df)}"
    )
    return df


if __name__ == "__main__":
    REAL_DATA_PATH = os.path.join("data", "raw", "train.csv")

    try:
        check_and_download_data(REAL_DATA_PATH)
    except Exception as e:
        logger.error(f"Ingestion process failed: {e}")
        exit(1)
