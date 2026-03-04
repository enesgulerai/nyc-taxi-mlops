import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Modern path yönetimi
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.components.feature_engineering import create_features


class TestFeatureEngineering:
    """
    Unit Tests for the Feature Engineering component.
    Tests whether the raw data can be transformed into the features (distance, hour, etc.) expected by the model.
    """

    @pytest.fixture
    def mock_raw_data(self) -> pd.DataFrame:
        """
        Generates dummy NYC Taxi data for testing purposes.
        """
        data = {
            # 1st Row: Tuesday, 2nd Row: Saturday
            "pickup_datetime": ["2026-01-20 10:00:00", "2026-01-24 10:00:00"],
            "dropoff_datetime": ["2026-01-20 10:15:00", "2026-01-24 10:20:00"],
            "passenger_count": [1, 2],
            "pickup_longitude": [-73.9857, -73.9857],
            "pickup_latitude": [40.7484, 40.7484],
            "dropoff_longitude": [-73.9665, -73.9665],
            "dropoff_latitude": [40.7812, 40.7812],
        }

        df = pd.DataFrame(data)

        # Datetime Transformation
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

        return df

    @pytest.fixture
    def processed_data(self, mock_raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the feature engineering function once per test to DRY up the code.
        Uses .copy() to prevent SettingWithCopyWarning and mutation side-effects.
        """
        return create_features(mock_raw_data.copy())

    def test_distance_calculation(self, processed_data: pd.DataFrame):
        """
        Test: Does 'distance_haversine' column exist and is it logical?
        """
        assert "distance_haversine" in processed_data.columns, (
            "Important feature: 'distance_haversine' was not created."
        )

        distance = processed_data["distance_haversine"].iloc[0]

        # Tip ve mantık kontrolü
        assert isinstance(distance, (float, np.floating)), "Distance must be a float."
        assert distance > 0, (
            "The distance was calculated as 0; check the Haversine formula."
        )

    def test_time_features_structure(self, processed_data: pd.DataFrame):
        """
        Test: Are the time-based attributes (hour, day_of_week) being named correctly?
        """
        required_features = ["hour", "day_of_week", "is_weekend"]

        for feature in required_features:
            assert feature in processed_data.columns, f"Feature '{feature}' is missing."

    def test_is_weekend_logic(self, processed_data: pd.DataFrame):
        """
        Test: Does the 'is_weekend' logic work correctly?
        """
        # Row 1: Tuesday, January 20, 2026 -> Weekday (should be 0)
        assert processed_data.iloc[0]["is_weekend"] == 0, (
            "Tuesday was marked as the weekend (ERROR)."
        )

        # Row 2: Saturday, January 24, 2026 -> Weekend (should be 1)
        assert processed_data.iloc[1]["is_weekend"] == 1, (
            "Saturday is marked as a weekday (ERROR)."
        )

    def test_output_not_empty(self, processed_data: pd.DataFrame):
        """
        Test: The output should not be blank and no rows should be dropped.
        """
        assert not processed_data.empty, "The processed data returned nothing."
        assert len(processed_data) == 2, (
            f"Expected 2 rows, but got {len(processed_data)}."
        )
