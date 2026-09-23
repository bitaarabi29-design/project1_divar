"""

آزمون فرض شماره 4

بررسی میزان تاثیر امکانات لاکچری و غیر لاکچری بر روی قیمت خانه 

"""
# وارد کردن کتابخانه ها 
import sys
import pandas as pd
import statsmodels.api as sm

#برای اینکه نوشته های فارسی درست باشه و خطا ندن
sys.stdout.reconfigure(encoding='utf-8')
#خواندن فایل
file_path = r"C:\Users\Asus\Desktop\final_data.csv"
#برای جلوگیری از هشدار نوع داده : low_memory=False
df = pd.read_csv(file_path, low_memory=False)



# فیلتر کردن آگهی‌های تهران و حذف قیمت‌های خالی
df_tehran = df[(df['city_slug'] == 'tehran') & (df['price_value'].notna())].copy()


#  مشخص کردن ستون‌های امکانات لاکچری و غیرلاکچری
luxury_cols = ['is_pool_tehran', 'is_sauna_tehran', 'is_jacuzzi_tehran', 'is_barbecue_tehran', 'is_security_guard_tehran']
non_luxury_cols = ['is_elevator_tehran', 'is_gas_tehran', 'is_water_tehran', 'is_electricity_tehran']

all_features = luxury_cols + non_luxury_cols

# تبدیل متغیرهای بولین به 0 و 1
for col in all_features:
    df_tehran[col] = df_tehran[col].astype(int)

#آماده‌سازی داده‌ها برای رگرسیون
X = df_tehran[all_features]
X = sm.add_constant(X)  # مقدار ثابت عرص از مبدا
Y = df_tehran['price_value']


#برازش مدل رگرسیون
model = sm.OLS(Y, X).fit()

#چاپ جدول خلاصه نتایج
print("\n=================== خلاصه نتایج آزمون فرض ===================")
print(model.summary())

# تحلیل هوشمند نتایج نهایی
print("\n=================== تحلیل نهایی نتایج ===================")
p_values = model.pvalues
coefs = model.params

for col in all_features:
    p_val = p_values[col]
    coef = coefs[col]
    is_significant = p_val < 0.05
    group = "لاکچری" if col in luxury_cols else "غیرلاکچری"
    
    status = "تاثیر معنادار دارد " if is_significant else "تاثیر معنادار ندارد "
    print(f"ویژگی ({group}) - {col}:")
    print(f"  - ضریب تغییر قیمت: {coef:,.0f} تومان")
    print(f"  - p-value برابر است با: {p_val:.4f} -> {status}\n")
"""
نتیجه :امکانات لاکچری (استخر و سونا و جکوزی و یاربیکیو و نگهبانی)
تاثیر مثبت و معناداری بر قیمت مسکن دارند، اما امکانات غیرلاکچری 
و پایه‌ای به دلیل همه‌گیر بودن، هیچ تاثیر معناداری بر تغییرات قیمت ندارند

"""