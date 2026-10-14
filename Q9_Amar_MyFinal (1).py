"""
سوال 9 آمار 
"""
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

def fix_fa(text):
    return get_display(arabic_reshaper.reshape(str(text)))

# خوندن دیتا
parquet_path = r"C:\Users\Asus\Downloads\divar_clean.parquet"
df = pd.read_parquet(parquet_path)

#ویژگی ها
features_info = {
    'has_balcony': {'name': 'بالکن', 'color': '#00A896'},
    'has_elevator': {'name': 'آسانسور', 'color': '#DDA15E'},
    'has_security_guard': {'name': 'نگهبان', 'color': '#9b59b6'},
    'has_barbecue': {'name': 'باربیکیو', 'color': '#f39c12'},
    'has_pool': {'name': 'استخر', 'color': '#3498db'}}

for col, info in features_info.items():
    # حذف داده‌های خالی 
    df_temp = df[['city_slug', col]].dropna()
    
    # محاسبه درصد خانه‌های دارای ویژگی
    city_stats = df_temp.groupby('city_slug')[col].agg(
        total='count',
        has_feat=lambda x: (x == True).sum())
    
    # فیلتر شهرهای با حداقل ۵۰ آگهی
    city_stats = city_stats[city_stats['total'] >= 50]
    
    city_stats['percent'] = (city_stats['has_feat'] / city_stats['total']) * 100
    
    #سی شهر برتر از نظر درصد
    top_30 = city_stats.sort_values(by='percent', ascending=False).head(30)
    
    plt.figure(figsize=(10, 10))
    cities_fa = [fix_fa(c) for c in top_30.index]
    
    # رسم نمودار افقی
    bars = plt.barh(cities_fa[::-1], top_30['percent'].values[::-1], color=info['color'])
    
    plt.title(fix_fa(f'۳۰ شهر برتر از نظر درصد خانه‌های دارای {info["name"]}'))
    plt.xlabel(fix_fa('درصد خانه‌ها (%)'))
    plt.ylabel(fix_fa('شهر'))
    plt.xlim(0, max(top_30['percent']) * 1.15)
    plt.grid(axis='x', alpha=0.3)
    
    # نمایش درصد روی میله‌ها
    for bar in bars:
        val = bar.get_width()
        plt.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8)
        
    plt.tight_layout()
    plt.show()