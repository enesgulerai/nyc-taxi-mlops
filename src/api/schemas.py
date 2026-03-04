from pydantic import BaseModel, ConfigDict, Field


# DATA COMING FROM THE USER
class TaxiInput(BaseModel):
    pickup_datetime: str  # "2026-01-18 14:30:00" | We are waiting in the format

    # Koordinat sınırlarını Field ile belirliyoruz (ge: greater or equal, le: less or equal)
    pickup_longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Boylam -180 ile 180 arasında olmalıdır."
    )
    pickup_latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Enlem -90 ile 90 arasında olmalıdır."
    )

    dropoff_longitude: float = Field(..., ge=-180.0, le=180.0)
    dropoff_latitude: float = Field(..., ge=-90.0, le=90.0)

    # Bir taksiye mantıken en az 1, en fazla 8-10 kişi binebilir
    passenger_count: int = Field(
        ..., ge=1, le=10, description="Yolcu sayısı 1 ile 10 arasında olmalıdır."
    )

    # Pydantic V2 standartlarına uygun modern Config yapısı
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
