# وارد کردن کتابخانه‌ها
import os
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# نمایش فارسی و راست به چپ
def fix_fa(text):
  reshaped_text = arabic_reshaper.reshape(str(text))
  return get_display(reshaped_text)


# خواندن فایل تمیز شده جدید و ستون های مورد نیاز
file_path = os.path.join(os.path.expanduser("~"), "Desktop", "final_data.csv")
columns = [
    "price_value",
    "land_size",
    "building_size",
    "regular_person_capacity",
    "rooms_count",
    "location_latitude",
    "location_longitude",
]

df = pd.read_csv(file_path, usecols=columns)

# اطمینان از اینکه ستون‌های عددی به صورت float یا int هستند
for col in columns:
  df[col] = pd.to_numeric(df[col], errors="coerce")

# ماتریس همبستگی
correlation_matrix = df[columns].corr(method="pearson")

# ترجمه ستون ها 
persian_labels = {
    "price_value": "مبلغ قیمت",
    "land_size": "متراژ زمین",
    "building_size": "زیربنا",
    "regular_person_capacity": "ظرفیت نفرات",
    "rooms_count": "تعداد اتاق‌ها",
    "location_latitude": "عرض جغرافیایی",
    "location_longitude": "طول جغرافیایی",
}

labels = [fix_fa(persian_labels[col]) for col in columns]

# اعداد داخل نمودار
annot_matrix = correlation_matrix.map(
    lambda x: "N/A" if pd.isna(x) else f"{x:.2f}"
)

# رنگ
cmap = plt.cm.coolwarm.copy()
cmap.set_bad("#BDBDBDC7")

# ایجاد شکل
fig, ax = plt.subplots(figsize=(14, 12))
fig.subplots_adjust(top=0.90, bottom=0.20, left=0.20, right=0.88)

# رسم
sns.heatmap(
    correlation_matrix,
    annot=annot_matrix,
    fmt="",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=1.5,
    linecolor="white",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"fontsize": 12, "fontweight": "bold", "color": "black"},
    cbar_kws={"shrink": 0.80},
    ax=ax,
)

# محور افقی
ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=12)

# محور عمودی
ax.set_yticklabels(labels, rotation=0, fontsize=12)

# فاصله برچسب‌ها
ax.tick_params(axis="x", pad=10)
ax.tick_params(axis="y", pad=10)

# عنوان
ax.set_title(
    fix_fa("ماتریس همبستگی متغیرها"), fontsize=18, fontweight="bold", pad=20
)

# نوار رنگ
cbar = ax.collections[0].colorbar
cbar.set_label(
    fix_fa("ضریب همبستگی"), fontsize=12, rotation=270, labelpad=20
)

# ذخیره و نمایش نمودار
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()