# وارد کردن کتابخانه ها
import os
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


# متن فارسی و راست به چپ
def fix_fa(text):
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)


# خواندن دیتای تمیز شده جدید
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "final_data.csv")
df = pd.read_csv(desktop_path, low_memory=False)

# تبدیل تاریخ به فرمت datetime برای استخراج سال و ماه
df['created_at_month'] = pd.to_datetime(df['created_at_month'], errors='coerce')


# تابع تبدیل تاریخ میلادی به سال شمسی
def gregorian_to_jalali_year(date):
    if pd.isna(date):
        return None

    year = date.year
    if year == 2024:
        start_date = pd.Timestamp('2024-03-20')
    else:
        start_date = pd.Timestamp(f'{year}-03-21')

    if date >= start_date:
        return year - 621
    else:
        return year - 622


# ساخت ستون year از روی تاریخ موجود در دیتای تمیز
df['year'] = df['created_at_month'].apply(gregorian_to_jalali_year)

# انتخاب سال های مورد نظر
df = df[df['year'].isin([1400, 1401, 1402, 1403])]

# نرخ تورم سالانه (داده ها از سایت رسمی بانک مرکزی)
inflation = {1400: 46.2, 1401: 53.1, 1402: 47.4, 1403: 35.8}

# شاخص قیمت (سال پایه 1400)
price_index = {1400: 100}
for year in [1401, 1402, 1403]:
  price_index[year] = price_index[year - 1] * (1 + inflation[year] / 100)

# میانگین قیمت هر سال
annual_price = df.groupby('year')['price_value'].mean().reset_index()

# گذاشتن قیمت در جدول
annual_price['price_index'] = annual_price['year'].map(price_index)

# محاسبه قیمت حقیقی ( سال پایه 1403)
base_year = 1403
annual_price['real_price'] = (
    annual_price['price_value'] * price_index[base_year]
    / annual_price['price_index'])


years = annual_price['year'].tolist()
year_labels = [fix_fa(f'سال {int(year)}') for year in years]


# رسم نمودار
plt.figure(figsize=(11, 6))
plt.plot(
    years,
    annual_price['real_price'],
    color='#4D0547',
    marker='o',
    markersize=8,
    linewidth=3.5,)

# فارسی کردن محور ایکس
plt.xticks(years, year_labels, fontsize=11)


# فارسی کردن اعداد محور ایگرگ
def format_y(value, position):
    return fix_fa(f'{value:,.0f}')


plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y))

# عنوان محور ها
plt.xlabel(fix_fa('سال'), fontsize=13)
plt.ylabel(fix_fa('قیمت حقیقی'), fontsize=13)
plt.title(
    fix_fa('روند قیمت حقیقی ملک در سال‌های ۱۴۰۰ تا ۱۴۰۳'),
    fontsize=15,
    fontweight='bold',)
plt.grid(True,linestyle='--', alpha=0.8, linewidth=1.5)
# نمایش
plt.tight_layout()
plt.show()