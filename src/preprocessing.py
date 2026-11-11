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

SERVICE_OPTIONS = ["No", "No internet service", "Yes"]
CATEGORY_OPTIONS = {
    "Multiple Lines": ["No", "No phone service", "Yes"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": SERVICE_OPTIONS,
    "Online Backup": SERVICE_OPTIONS,
    "Device Protection": SERVICE_OPTIONS,
    "Tech Support": SERVICE_OPTIONS,
    "Streaming TV": SERVICE_OPTIONS,
    "Streaming Movies": SERVICE_OPTIONS,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Payment Method": [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check",
    ],
}
CHARGE_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges"]
RAW_COLUMNS = {*CHARGE_COLUMNS, "Gender", "Senior Citizen", *YES_NO_COLUMNS, *CATEGORY_OPTIONS}


def encode_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    tenure = pd.to_numeric(df["Tenure Months"], errors="raise")
    total = pd.to_numeric(df["Total Charges"], errors="coerce")
    empty_total = df["Total Charges"].isna() | df["Total Charges"].astype(str).str.strip().eq("")
    total = total.mask(empty_total & tenure.eq(0), 0.0)
    df["Tenure Months"] = tenure
    df["Monthly Charges"] = pd.to_numeric(df["Monthly Charges"], errors="raise")
    df["Total Charges"] = total
    if not (df[CHARGE_COLUMNS].ge(0) & df[CHARGE_COLUMNS].lt(float("inf"))).all().all():
        raise ValueError("Tenure and charges must be finite and non-negative")
    if not tenure.mod(1).eq(0).all():
        raise ValueError("Tenure Months must be an integer")

    mappings = {
        "Gender": {"Male": 1, "Female": 0},
        "Senior Citizen": {"Yes": 1, "No": 0, "1": 1, "0": 0},
        **{column: {"Yes": 1, "No": 0} for column in YES_NO_COLUMNS},
    }
    for column, mapping in mappings.items():
        values = df[column].astype(str).str.strip().map(mapping)
        if values.isna().any():
            raise ValueError(f"Invalid value for {column}")
        df[column] = values.astype(int)

    for column, options in CATEGORY_OPTIONS.items():
        values = pd.Categorical(df[column], categories=options)
        if values.isna().any():
            raise ValueError(f"Invalid value for {column}")
        df[column] = values
    return pd.get_dummies(df, columns=list(CATEGORY_OPTIONS), drop_first=True, dtype=int)


def preprocess():
    df = encode_customers(load_data())
    save_path = BASE_DIR / "data" / "v2" / "clean_data.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    print(df.info())
    print(df.shape)
    print(f"\nSaved -> {save_path}")

    return df


if __name__ == "__main__":
    preprocess()
