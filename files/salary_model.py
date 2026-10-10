import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "ds_salaries.csv"
MODEL_PATH = BASE_DIR / "salary_model.joblib"
TARGET = "salary_in_usd"
CATEGORICAL = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
]
NUMERIC = ["work_year", "remote_ratio"]


def load(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "salary", "salary_currency"], errors="ignore")
    df["same_country"] = (df["employee_residence"] == df["company_location"]).astype(int)
    return df


def build(estimator):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3), CATEGORICAL),
            ("num", "passthrough", NUMERIC + ["same_country"]),
        ]
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("est", estimator)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return model


def main():
    df = load(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "ridge": RidgeCV(alphas=np.logspace(-2, 3, 20)),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    results = {}
    for name, est in candidates.items():
        scores = cross_validate(build(est), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "r2": scores["test_r2"].mean(),
            "mae": -scores["test_mae"].mean(),
            "rmse": -scores["test_rmse"].mean(),
        }
    cv_table = pd.DataFrame(results).T.sort_values("rmse")
    print("5-fold CV on train split")
    print(cv_table.round(3).to_string())

    best_name = cv_table.index[0]
    best = build(candidates[best_name]).fit(X_train, y_train)
    pred = best.predict(X_test)
    print(f"\nBest: {best_name}")
    print(f"Test R2:   {r2_score(y_test, pred):.3f}")
    print(f"Test MAE:  ${mean_absolute_error(y_test, pred):,.0f}")
    print(f"Test RMSE: ${root_mean_squared_error(y_test, pred):,.0f}")

    final = build(candidates[best_name]).fit(X, y)
    joblib.dump(final, MODEL_PATH)


def predict(**row):
    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([row])
    frame["same_country"] = (frame["employee_residence"] == frame["company_location"]).astype(int)
    return float(model.predict(frame)[0])


if __name__ == "__main__":
    main()