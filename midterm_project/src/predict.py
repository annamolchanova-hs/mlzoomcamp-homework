import json
import requests


if __name__ == '__main__':
    url = 'http://localhost:9696/predict'
    record = {
        'study_hours_per_day': 2,
        'extracurricular_hours_per_day': 6,
        'sleep_hours_per_day': 9,
        'social_hours_per_day': 1,
        'physical_activity_hours_per_day': 1,
        'stress_level': 'Low',
    }
    response = requests.post(url, json=record)
    print(response.text)
