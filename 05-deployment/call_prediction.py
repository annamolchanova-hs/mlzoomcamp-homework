import requests


if __name__ == '__main__':
    url = "http://0.0.0.0:9696/predict"
    client = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0
    }
    print(requests.post(url, json=client).json())
