"""
آزمون فرض 4
"""
# وارد کردن کتابخانه ها
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
from scipy.stats import mannwhitneyu
import zipfile

# خواندن دیتا از فایل زیپ
zip_path = "data/final_data_with_parking_text.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('final_data_with_parking_text.csv') as f:
        df = pd.read_csv(f, low_memory=False)

# فقط تهران
df = df[df["city_slug"] == "tehran"].copy()

# فقط فروش
df = df[df["deal_family"] == "sell"].copy()

# قیمت رو تمیز می‌کنیم
price = "price_value"
df = df[df[price].notna() & (df[price] > 0)]
low, high = df[price].quantile([0.01, 0.99])
df = df[(df[price] >= low) & (df[price] <= high)]

# تابع آزمون من ویتنی
def mw_test(df, col, name, price_col=price):
    sub = df[df[col].notna()]
    g1 = sub.loc[sub[col] == True, price_col]
    g0 = sub.loc[sub[col] == False, price_col]

    # آزمون من ویتنی
    u, p = mannwhitneyu(g1, g0, alternative="two-sided")

    # میانه و درصد تفاوت
    med1, med0 = g1.median(), g0.median()
    diff = (med1 - med0) / med0 * 100

    # چاپ نتایج آزمون
    print(f"\n{name}")
    print(f"  n_with    = {len(g1):,}")
    print(f"  n_without = {len(g0):,}")
    print(f"  med_with    = {med1:,.0f}")
    print(f"  med_without = {med0:,.0f}")
    print(f"  diff = {diff:+.1f}%")
    print(f"  p    = {p:.3g}")

    # تفسیر خودکار
    if p < 0.05:
        if abs(diff) >= 100:
            print("  نتیجه: تفاوت معنادار و بزرگ")
        elif abs(diff) >= 30:
            print("  نتیجه: تفاوت معنادار و متوسط")
        elif abs(diff) >= 10:
            print("  نتیجه: تفاوت معنادار ولی کوچک")
        else:
            print("  نتیجه: تفاوت معنادار ولی ناچیز")
    else:
        print("  نتیجه: تفاوت غیرمعنادار")
    print()

# آزمون استخر
mw_test(df, "is_pool_tehran", "استخر (لاکچری)")

# آزمون پارکینگ
mw_test(df, "is_parking_tehran", "پارکینگ (غیرلاکچری)")
