from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def create_features():
    data_path = BASE_DIR / "data" / "v2" / "clean_data.csv"
    df = pd.read_csv(data_path)

    # ===========================================
    # Feature 1: monthly cost's means
    # ===========================================
    df["AvgChargePerMonth"] = df["Total Charges"] / (df["Tenure Months"] + 1)

    # ===========================================
    # Feature 2: customer membership duration group
    # ===========================================
    df["TenureGroup"] = pd.cut(
        df["Tenure Months"], bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3], include_lowest=True,
    ).astype(int)

    # ===========================================
    # Feature 3:valued customer depend on Monthly Charges
    # ===========================================
    df["HighValueCustomer"] = (df["Monthly Charges"] > df["Monthly Charges"].median()).astype(int)

    # ===========================================
    # Feature 4: long-term contract customer
    # ===========================================
    if "Contract_One year" in df.columns and "Contract_Two year" in df.columns:
        df["LongContract"] = df["Contract_One year"] + df["Contract_Two year"]

    save_path = BASE_DIR / "data" / "v3" / "feature_data.csv"
    df.to_csv(save_path, index=False)

    print("=" * 60)
    print("Feature Engineering Finished")
    print("=" * 60)
    print(df.info())
    print(df.shape)
    print("\nSaved to", save_path)

    return df


if __name__ == "__main__":
    create_features()
