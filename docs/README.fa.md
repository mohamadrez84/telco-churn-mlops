# پیش‌بینی ریزش مشتریان مخابرات

این پروژه چهار مدل را روی دادهٔ نمونهٔ IBM و دو مجموعه ویژگی مقایسه می‌کند. مدل منتخب XGBoost روی v2 است. در تقسیم داخلی، Accuracy از ۷۶٫۱۶٪ به ۸۰٫۲۳٪ رسید و F1 ریزش از ۰٫۶۲۸۳ به ۰٫۶۲۰۷ تغییر کرد. مدل جدید هشدار اشتباه کمتری می‌دهد و موارد ریزش کمتری را شناسایی می‌کند. [گزارش روش‌شناسی](model-report.md) و [جدول نتایج](../results/model_comparison.csv) جزئیات مقایسه را دارند.

## اجرا

دستورها را از ریشهٔ پروژه با Python 3.13 اجرا کنید.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m run_pipeline
mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db
```

برای اجرای API، دستور `python -m deployment.app` را اجرا کنید. مسیر `/health` وضعیت سرویس را گزارش می‌کند. برای `/predict` از [نمونهٔ JSON](../examples/customer.json) استفاده کنید. راهنمای Docker در [صفحهٔ اصلی](../README.md#run-the-api) آمده است.

## ارزیابی

پارامترهای پیش‌پردازش در هر fold روی بخش آموزش همان fold محاسبه می‌شوند. آستانه‌های تصمیم از پیش‌بینی‌های خارج از fold پیشنهاد می‌شوند. مدل با بیشترین Accuracy اعتبارسنجی انتخاب می‌شود، به شرط آنکه F1 آن حداکثر ۰٫۰۱ از بهترین مدل پیش‌فرض کمتر باشد. در اجرای جدید فقط مدل منتخب روی Test ارزیابی می‌شود. برای سنجش روی دادهٔ جداگانه، فایل CSV یا XLSX برچسب‌دار را به `python -m evaluate_external` بدهید. مجموعهٔ مستقل برای گزارش امتیاز بیرونی در اختیار پروژه نبوده است.

[خلاصهٔ آزمایش‌ها](../results/experiment_summary.json) و [ماتریس درهم‌ریختگی](../results/confusion_matrix.png) مستقیماً از GitHub قابل مشاهده‌اند. برای انتقال تاریخچهٔ کامل MLflow، [راهنمای سرور و فایل‌های خروجی](model-report.md#mlflow-on-another-machine) را ببینید.
