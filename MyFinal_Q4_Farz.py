"""
آزمون فرض 4
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from scipy.stats import mannwhitneyu

path = r"C:\Users\Asus\Downloads\divar_clean.parquet"
df = pd.read_parquet(path)

# فقط مسکونی
if 'is_residential_unit' in df.columns:
    df = df[df['is_residential_unit'] == True]

p_col ='price_value'

df = df[df[p_col].notna() & (df[p_col] > 0)]

# حذف پرت ها
q1 = df[p_col].quantile(0.05)
q2 = df[p_col].quantile(0.95)

df = df[(df[p_col] >= q1) & (df[p_col] <= q2)].copy()

lux_cols = ['has_pool_combined','has_barbecue_combined','has_sauna_combined','has_jacuzzi_combined']

non_lux_cols = ['has_parking','has_elevator','has_balcony_combined']
for c in lux_cols + non_lux_cols:
    if c in df.columns:
        df[c] = df[c].fillna(False).astype(bool)

# حداقل یکی
lux = [c for c in lux_cols if c in df.columns]
non_lux = [c for c in non_lux_cols if c in df.columns]
df['has_luxury'] = df[lux].any(axis=1)
df['has_non_luxury'] = df[non_lux].any(axis=1)


def test(data, col, name):

    a = data[data[col] == True][p_col]
    b = data[data[col] == False][p_col]

    u, p = mannwhitneyu(a, b, alternative='two-sided')

    med_a = a.median()
    med_b = b.median()

    diff = ((med_a - med_b) / med_b) * 100

    print(f"=== {name} ===")
    print(f"تعداد با امکانات: {len(a):,}")
    print(f"تعداد بدون امکانات: {len(b):,}")
    print(f"میانه قیمت با امکانات: {med_a:,.0f}")
    print(f"میانه قیمت بدون امکانات: {med_b:,.0f}")
    print(f"درصد تغییر میانه: {diff:+.2f}%")
    print(f"p-value: {p}\n")


test(df,'has_luxury','امکانات لاکچری (استخر، باربیکیو، سونا، جکوزی)')
test(df,'has_non_luxury','امکانات غیرلاکچری (پارکینگ، آسانسور، بالکن)')