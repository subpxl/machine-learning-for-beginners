import asyncio
import time
import httpx

URL = "http://127.0.0.1:8000/predict"

sample_data = {
    "Time": 0,
    "V1": -1.35,
    "V2": -0.07,
    "V3": 2.53,
    "V4": 1.37,
    "V5": -0.33,
    "V6": 0.46,
    "V7": 0.23,
    "V8": 0.09,
    "V9": 0.36,
    "V10": 0.09,
    "V11": -0.55,
    "V12": -0.61,
    "V13": -0.99,
    "V14": -0.31,
    "V15": 1.46,
    "V16": -0.47,
    "V17": 0.20,
    "V18": 0.02,
    "V19": 0.40,
    "V20": 0.25,
    "V21": -0.01,
    "V22": 0.27,
    "V23": -0.11,
    "V24": 0.06,
    "V25": 0.12,
    "V26": -0.18,
    "V27": 0.13,
    "V28": -0.02,
    "Amount": 149.62
}

TOTAL_REQUESTS = 1000


async def send_request(client, request_id):

    response = await client.post(URL, json=sample_data)

    return response.status_code


async def main():

    start = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:

        tasks = [
            send_request(client, i)
            for i in range(TOTAL_REQUESTS)
        ]

        results = await asyncio.gather(*tasks)

    end = time.time()

    success = sum(1 for r in results if r == 200)

    print(f"Total Requests: {TOTAL_REQUESTS}")
    print(f"Successful Requests: {success}")
    print(f"Total Time: {end-start:.2f} sec")

    print(f"Requests/sec: {TOTAL_REQUESTS/(end-start):.2f}")


if __name__ == "__main__":
    asyncio.run(main())