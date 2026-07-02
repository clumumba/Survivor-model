from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


import pathlib, sys, joblib

#DATA_PATH = Path(__file__).resolve().parent / "data" / "tested.csv"
_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

Data_path = "tested.csv"
TARGET = "Survived"
FEATURES = ["Age", "Sex", "Pclass"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(
        columns=["Name", "Ticket", "Cabin", "PassengerId", "Embarked"],
        errors="ignore",
    )
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    return df


def main() -> None:
    data = pd.read_csv(Data_path)
    data = clean_data(data)

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = Pipeline([
            ("scaler", MinMaxScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    main()

