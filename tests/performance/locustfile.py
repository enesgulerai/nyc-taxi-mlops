import random
from datetime import datetime, timedelta

from locust import HttpUser, between, task


class TaxiUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predict_duration(self):
        random_days = random.randint(0, 30)
        pickup_time = datetime.now() + timedelta(days=random_days)

        payload = {
            "pickup_datetime": pickup_time.strftime("%Y-%m-%d %H:%M:%S"),
            "pickup_longitude": -73.985 + random.uniform(-0.05, 0.05),
            "pickup_latitude": 40.748 + random.uniform(-0.05, 0.05),
            "dropoff_longitude": -73.985 + random.uniform(-0.05, 0.05),
            "dropoff_latitude": 40.748 + random.uniform(-0.05, 0.05),
            "passenger_count": random.randint(1, 6),
        }

        with self.client.post(
            "/predict", json=payload, catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predicted_duration_minutes" in data:
                        response.success()
                    else:
                        response.failure(f"Missing key in response: {data.keys()}")
                except Exception as e:
                    response.failure(f"JSON Decode Error: {e}")

            elif response.status_code == 422:
                response.failure(f"Validation Error (Check Schema): {response.text}")

            elif response.status_code == 503:
                response.failure("Model Service Not Ready")

            else:
                response.failure(f"Status code: {response.status_code}")
