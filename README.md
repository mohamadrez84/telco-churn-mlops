# Customer Churn MLOps
## نکات مهم پروژه

|---|---|---|
| `src/data_loader.py` | مسیر مطلق ویندوزی → مسیر نسبی با `Path` | قابل حمل به هر سیستم/Docker |

| `src/preprocessing.py` | حذف `CLTV`, `Latitude`, `Longitude` | جلوگیری از Data Leakage |

| `src/preprocessing.py` | پرکردن `Total Charges` با ۰ به‌جای median | منطق کسب‌وکاری درست‌تر (Tenure=0) |

| `src/features.py` | حذف فیچر `CustomerValueIndex` (وابسته به CLTV) | چون CLTV دیگر وجود ندارد |

| `src/train.py` | اضافه‌شدن split سه‌گانه Train/Val/Test | جلوگیری از انتخاب مدل با Test |

| `src/train.py` | اضافه‌شدن `StandardScaler` (fit فقط روی Train) | جلوگیری از نشتی در نرمال‌سازی |

| `src/train.py` | مقایسه v2 **و** v3 (قبلاً فقط v3) | سنجش واقعی اثر Feature Engineering |

| `src/evaluate.py` | فایل جدید | جدا کردن منطق ارزیابی از آموزش |

| `src/mlflow_utils.py` | فایل جدید | ثبت کامل‌تر (Confusion Matrix + خود مدل) |

| `run_pipeline.py` | فایل جدید | یک دستور برای اجرای کل فرآیند |

| `Dockerfile` + `deployment/app.py` | فایل جدید | مرحله ۴ مستند پروژه (Deployment) |

| `requirements.txt` | فایل جدید | تکرارپذیری نصب محیط |

## نحوه اجرا

```bash
pip install -r requirements.txt
python run_pipeline.py
```

بعد از اجرا:
- `data/v2/clean_data.csv` و `data/v3/feature_data.csv` بازتولید می‌شوند
- `mlflow.db` با ۸ Run (۴ مدل × ۲ نسخه) پر می‌شود
- `results/model_comparison.csv` جدول مقایسه کامل را نشان می‌دهد
- `models/best_model.pkl` بهترین مدل (بر اساس F1 روی Test) را نگه می‌دارد

### دیدن نتایج در MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### استقرار با Docker
```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"Tenure Months": 12, ...}'
```

## نکته مهم درباره نتیجه واقعی این اجرا

بعد از رفع نشتی‌ها، F1-score واقعی حدود **0.63-0.65** است (نه بیشتر) —
این عدد واقعی است. اگر CLTV را نگه می‌داشتیم عدد بالاتری
می‌گرفتیم اما آن عدد دروغ بود (مدل در عمل آن‌قدر خوب کار نمی‌کرد).
