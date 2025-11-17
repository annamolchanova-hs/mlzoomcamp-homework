import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

import xgboost as xgb
from sklearn.metrics import root_mean_squared_error


SEED = 42

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': SEED,
    'verbosity': 1,

    # tuned parameters
    'booster': 'gblinear',
    'learning_rate': 1,
    'n_estimators': 20,
}


def prepare_data():
    df = pd.read_csv('data/study_habits.csv')

    del df['Student_ID']
    df.columns = df.columns.str.lower()
    X = df.copy()
    del X['gpa']
    y = df['gpa']

    numerical_features = list(X.select_dtypes(exclude=["object"]).columns)
    categorical_features = list(X.select_dtypes(include=["object"]).columns)

    num_scaler = StandardScaler()
    cat_encoder = OneHotEncoder(handle_unknown='error')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_scaler, numerical_features),
            ('cat', cat_encoder, categorical_features),
        ],
        remainder='passthrough'  # Keep other columns as they are (if any)
    )

    X = preprocessor.fit_transform(X)
    feature_names = [f.lower() for f in preprocessor.get_feature_names_out()]
    X = pd.DataFrame(X, columns=feature_names)

    return X, y, preprocessor


def train_xgb():
    X, y, preprocessor = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = xgb.XGBRegressor(**XGB_PARAMS)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    RMSE = root_mean_squared_error(y_test, y_pred)
    print(f'{RMSE=}')

    with open('artifacts/xgb_model.pkl', 'wb') as f:
        pickle.dump((preprocessor, model), f)


if __name__ == "__main__":
    train_xgb()