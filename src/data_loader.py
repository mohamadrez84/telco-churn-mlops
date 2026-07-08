from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_XLSX_PATH = BASE_DIR / "data" / "v1" / "Telco_customer_churn.xlsx"


def load_data(path: Path = RAW_XLSX_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"row dataset's file not found: {path}")
    df = pd.read_excel(path)
    return df


if __name__ == "__main__":
    data = load_data()
    print(data.shape)
    print(data.head())
