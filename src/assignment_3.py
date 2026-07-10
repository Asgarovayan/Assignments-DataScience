import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'Transported' column from Boolean
    (True/False) to Integer (1/0).

    Should return the modified DataFrame.
    """
    df["Transported"] = df["Transported"].astype(int)

    return df


def split_data(
    df: pd.DataFrame,
    features: list,
    target: str,
    test_size=0.2,
    random_state=42
):
    """
    Split the data into training and testing sets.

    - features: List of column names to use as features.
    - target: The name of the target column.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Pass an int for reproducible output.

    Returns: X_train, X_test, y_train, y_test
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> DecisionTreeClassifier:
    """
    Initialize and train a DecisionTreeClassifier
    on the training data.

    Return the trained model.
    """
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Predict the labels for the test set and calculate metrics.

    Returns a dictionary with 'accuracy' and 'f1_score'.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    return metrics


if __name__ == "__main__":
    print("Assignment 3 template ready.")
