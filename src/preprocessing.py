from pathlib import Path
import pandas as pd

from src.data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent.parent

DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",        
    "Longitude",       
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CLTV",
]

YES_NO_COLUMNS = ["Partner", "Dependents", "Phone Service", "Paperless Billing"]

CATEGORICAL_COLUMNS = [
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Payment Method",
]


def preprocess():
    df = load_data()

    # -------------------------
    # delete unusefull columns
    # -------------------------
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[preprocessing] {len(cols_to_drop)} column deleted : {cols_to_drop}")

    # -------------------------
    # Total Charges (missing values)
    # -------------------------
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    n_missing = df["Total Charges"].isna().sum()
    df["Total Charges"] = df["Total Charges"].fillna(0.0)
    print(f"[preprocessing] {n_missing} missing value in Total Charges filled with zero"
          f"(customer with Tenure Months=0).")

    # -------------------------
    # Gender
    # -------------------------
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # -------------------------
    # Senior Citizen
    # -------------------------
    df["Senior Citizen"] = (
        df["Senior Citizen"].astype(str).str.strip()
        .replace({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype(int)
    )

    # -------------------------
    # Yes / No
    # -------------------------
    for column in YES_NO_COLUMNS:
        df[column] = df[column].map({"Yes": 1, "No": 0})

    # -------------------------
    # Categorical Encoding (One-Hot)
    # -------------------------
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)

    save_path = BASE_DIR / "data" / "v2" / "clean_data.csv"
    df.to_csv(save_path, index=False)

    print(df.info())
    print(df.shape)
    print(f"\nSaved -> {save_path}")

    return df


if __name__ == "__main__":
    preprocess()
