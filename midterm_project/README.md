### Problem statement
The objective of this project is to examine the correlation between the daily lifestyles of students and their academic success. The dataset under investigation comprises information on a range of student activities, including study hours, sleep duration and physical activity, as well as stress levels. GPA (Grade Point Average) is a measure of academic performance.

This research generally analyses how time management, social and studying life balance, and mental well-being can affect academic performance. It ultimately allows for the prediction of students' GPA depending on their daily habits.

Project approach:
- target variable - GPA
- modeling method - models can be used to predict continuous value (Linear Regression, Random Forests Regression, XGBoost Regression),
- model performance evaluation metric - RMSE

### Dataset description
.CSV file for the project is stored as __data/study_habits.csv__. 
It contains 2000 rows, the schema is following:

| Column name                     | Column type | Description                                                                                  |
|:--------------------------------|:-----------:|:---------------------------------------------------------------------------------------------|
| Student_ID                      |     int     | A unique identifier assigned to each student.                                                |
| Study_Hours_Per_Day             |    float    | Average number of hours in which a student spends time for studying daily.                   |
| Extracurricular_Hours_Per_Day   |    float    | Spending time on extra-cocurricular activities such as clubs, arts,sports, or other hobbies. |
| Sleep_Hours_Per_Day             |    float    | Number of hours a student sleeps per day.                                                    |
| Social_Hours_Per_Day            |    float    | Time spent with friends, family, or social interactions.                                     |
| Physical_Activity_Hours_Per_Day |    float    | Time spent in physical activities or exercise.                                               |
| GPA                             |    float    | Grade Point Average representing academic performance.                                       |                                      |
| Stress_Level                    |   string    | Stress category of the student (Low, Moderate, High).                                        |

### EDA summary
Detailed analysis can be found in __notebooks/study_habits_eda.ipynb__. In general, data is very clean, without missing values or outliers.

### Modeling approach & metrics
__notebooks/study_habits_modeling.ipynb__ includes feature engineering and model selection process.

Feature engineering:
- scale numeric features
- apply one-hot encoding to categorical features

Trained and tuned models:
- Linear regression
- Linear regression with regularization
- Random forest regressor
- XGBoost

RMSE was used as model performance metric. According evaluations, the best model is XGBoost with parameters: {'booster': 'gblinear', 'learning_rate': 1, 'n_estimators': 20}

### Python scripts
`src/train.py` - logic for feature engineering and selected model training. Generated model and features prepocessor are stored in `artifacts` folder.  
`src/serve.py` - web service to make GPA predictions.  
`src/predict.py` - script to run an example prediction.

### Run locally
Poetry is used for project environment management.  
Pre-requirements:
```
python=3.11
poetry=1.8.4
```
To set up environment run in the project root dir:
```
poetry install
```
FastAPI web service start:
```
poetry run uvicorn src.serve:app --host 0.0.0.0 --port 9696
```
Run example prediction:
```
poetry run python src/predict.py
```
### Run via Docker
Build docker image:
```
 docker build -t study-habit .
```
Run predictions service via docker:
```
docker run -it -p 9696:9696 --rm study-habit:latest
```
### API usage example
```
curl -X POST -H "Content-Type: application/json" \
     -d '{"study_hours_per_day": 2, "extracurricular_hours_per_day": 6, "sleep_hours_per_day": 9,"social_hours_per_day": 1, "physical_activity_hours_per_day": 1, "stress_level": "Low"}' \
     http://localhost:9696/predict
```
```
{"prediction":2.387843370437622}
```