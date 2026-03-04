import pandas as pd

from src.utils.geo_utils import (
    calculate_bearing,
    dummy_manhattan_distance,
    haversine_array,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """It takes a raw dataframe and returns a dataframe with added features."""

    df = df.copy()

    # DATETIME CONVERSION
    if df["pickup_datetime"].dtype == "object":
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["month"] = df["pickup_datetime"].dt.month
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["hour"] = df["pickup_datetime"].dt.hour
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # GEOGRAPHIC DATA
    df["distance_haversine"] = haversine_array(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["distance_manhattan"] = dummy_manhattan_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["bearing"] = calculate_bearing(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    return df
