# اضافه کردن کتابخانه ها 
import arabic_reshaper
import jdatetime
import matplotlib.pyplot as plt
import pandas as pd
import os

# از راست به چپ کردن متن فارسی
from bidi.algorithm import get_display
plt.rcParams['font.family'] = 'Tahoma'

def fix_fa(text):
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)

# خواندن دیتای تمیز شده جدید
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "final_data.csv")
df_rent = pd.read_csv(desktop_path)

# اطمینان از اینکه ستون تاریخ به فرمت datetime تبدیل شده است (برای مرتب‌سازی صحیح)
df_rent['date_dt'] = pd.to_datetime(df_rent['created_at_month'])

# حذف داده‌های پرت و اشتباه (حذف اجاره‌های بالای ۵۰۰ میلیون تومان یا زیر ۱۰ هزار تومان)
df_rent = df_rent[(df_rent['rent_value'] >= 10000) & (df_rent['rent_value'] <= 500000000)]

# محاسبه میانه‌ اجاره هر ماه (استفاده از median به جای mean جهت جلوگیری از خراب شدن نمودار با داده‌های پرت)
trend_df = (df_rent
    .groupby(['date_dt', 'created_at_month'])['rent_value']
    .median()
    .reset_index())

# مرتب کردن تاریخ به ترتیب زمانی
trend_df = trend_df.sort_values('date_dt')

# تبدیل میلادی به شمسی
def to_jalali_month(dt):
    j_date = jdatetime.date.fromgregorian(date=dt.date())
    months_fa = [
        'فروردین', 'اردیبهشت', 'خرداد', 'تیر',
        'مرداد', 'شهریور', 'مهر', 'آبان',
        'آذر', 'دی', 'بهمن', 'اسفند'
    ]
    # متن مناسب نمایش در نمودار
    raw_str = f'{months_fa[j_date.month - 1]} {j_date.year}'
    return fix_fa(raw_str)

# ساختن ستون ماه شمسی 
trend_df['jalali_month'] = trend_df['date_dt'].apply(to_jalali_month)

# تعیین اندازه و رسم نمودار خطی 
plt.figure(figsize=(14, 6))
plt.plot(
    trend_df['jalali_month'],
    trend_df['rent_value'],
    marker='o',
    linestyle='-',
    color='#761314',
    linewidth=2.5
)

# تعیین عنوان نمودار
plt.title(
    fix_fa('ترند میانه قیمت اجاره بر حسب ماه‌های شمسی'),
    fontsize=15,
    pad=15
)

# اسامی محورها
plt.xlabel(fix_fa('ماه ثبت آگهی'), fontsize=12)
plt.ylabel(fix_fa('میانه قیمت اجاره (تومان)'), fontsize=12)

# مرتب کردن فاصله ها
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# نمایش
plt.show()