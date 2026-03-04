from pydantic import BaseModel, ConfigDict, Field


# DATA COMING FROM THE USER
class TaxiInput(BaseModel):
    pickup_datetime: str  # "2026-01-18 14:30:00" | We are waiting in the format

    pickup_longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="The longitude should be between -180 and 180 degrees."
    )
    pickup_latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="The latitude should be between -90 and 90 degrees."
    )

    dropoff_longitude: float = Field(..., ge=-180.0, le=180.0)
    dropoff_latitude: float = Field(..., ge=-90.0, le=90.0)

    passenger_count: int = Field(
        ..., ge=1, le=10, description="The number of passengers should be between 1 and 10."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pickup_datetime": "2026-01-18 14:30:00",
                "pickup_longitude": -73.985,
                "pickup_latitude": 40.758,
                "dropoff_longitude": -73.996,
                "dropoff_latitude": 40.732,
                "passenger_count": 1,
            }
        }
    )


class PredictionOutput(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
