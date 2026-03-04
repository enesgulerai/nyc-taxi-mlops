import asyncio
import time

import aiohttp

# SETTINGS
URL = "http://localhost:8000/predict"
TOTAL_REQUESTS = 10000
CONCURRENT_LIMIT = 100


def generate_payload():
    """
    CONSTANT DATA (FOR REDIS TESTING)
    We send the same coordinates every time.
    This way, the API will get the response from Redis (Cache), not the Model.
    """
    return {
        "passenger_count": 2,
        "pickup_longitude": -73.9857,
        "pickup_latitude": 40.7484,
        "dropoff_longitude": -73.9665,
        "dropoff_latitude": 40.7812,
        "pickup_datetime": "2026-01-20 15:30:00",
    }


async def send_request(session, request_id):
    payload = generate_payload()
    try:
        start_time = time.time()
        async with session.post(URL, json=payload) as response:
            await response.text()

            duration = (time.time() - start_time) * 1000
            status = response.status

            if status != 200:
                print(f"⚠️ Request {request_id} Failed: {status}")

            return status, duration
    except Exception as e:
        print(f"❌ Error on {request_id}: {e}")
        return 0, 0


async def main():
    print("🚀 REDIS IS OPENING....")
    print(f"🌊 Aim: {TOTAL_REQUESTS} request ({CONCURRENT_LIMIT} parallel connection)")
    print("-" * 50)

    # We are setting the TCP connection limit.
    connector = aiohttp.TCPConnector(limit=CONCURRENT_LIMIT)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        start_time = time.time()

        # Line up the requests
        for i in range(TOTAL_REQUESTS):
            task = asyncio.ensure_future(send_request(session, i))
            tasks.append(task)

        # Send them all
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    success_count = sum(1 for status, _ in results if status == 200)

    # Average Latency calculation
    if results:
        avg_latency = sum(d for _, d in results) / len(results)
    else:
        avg_latency = 0

    # RPS (Requests Per Second)
    rps = TOTAL_REQUESTS / total_time

    print("-" * 50)
    print("🏁 TEST COMPLETE! (REDIS RESULTS)")
    print(f"⏱️ Total Duration: {total_time:.2f} seconds")
    print(f"✅ Successful Request: {success_count}/{TOTAL_REQUESTS}")
    print(f"⚡ Average Latency (Client): {avg_latency:.2f} ms")
    print(f"🔥 RPS (Speed): {rps:.2f} requests/second")
    print("-" * 50)


if __name__ == "__main__":
    if (
        asyncio.get_event_loop_policy().__class__.__name__
        == "WindowsProactorEventLoopPolicy"
    ):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
