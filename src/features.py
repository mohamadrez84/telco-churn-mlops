from pathlib import Path
import pandas as pd

from src.preprocessing import encode_customers

BASE_DIR = Path(__file__).resolve().parent.parent


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AvgChargePerMonth"] = df["Total Charges"] / (df["Tenure Months"] + 1)
    df["TenureGroup"] = pd.cut(
        df["Tenure Months"], bins=[0, 12, 24, 48, float("inf")],
        labels=[0, 1, 2, 3], include_lowest=True,
    ).astype(int)
    df["LongContract"] = df["Contract_One year"] + df["Contract_Two year"]
    return df


def create_features(df=None):
    if df is None:
        df = pd.read_csv(BASE_DIR / "data" / "v2" / "clean_data.csv")
    df = add_features(df)
    save_path = BASE_DIR / "data" / "v3" / "feature_data.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    print("=" * 60)
    print("Feature Engineering Finished")
    print("=" * 60)
    print(df.info())
    print(df.shape)
    print("\nSaved to", save_path)

    return df


def transform_features(X, scaler, threshold):
    X = X.copy()
    if threshold is not None:
        X["HighValueCustomer"] = (X["Monthly Charges"] > threshold).astype(int)
    columns = list(scaler.feature_names_in_)
    X[columns] = scaler.transform(X[columns])
    return X


def prepare_raw_features(customers, bundle):
    """Apply the saved input transformations to raw customer records."""
    df = encode_customers(customers)
    is_v3 = bundle.get("dataset_version") == "v3" or "HighValueCustomer" in bundle["feature_columns"]
    if is_v3:
        if "high_value_threshold" not in bundle:
            raise ValueError("This legacy model requires a complete set of encoded features")
        df = add_features(df)
    df = transform_features(df, bundle["scaler"], bundle.get("high_value_threshold"))
    return df[bundle["feature_columns"]]


if __name__ == "__main__":
    create_features()
