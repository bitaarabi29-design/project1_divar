import os
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

#  فارسی
def fix_fa(text):
    return get_display(arabic_reshaper.reshape(str(text)))

# خوندن دیتا
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divar_clean.parquet")
columns = ["price_value", "land_size", "building_size",
    "regular_person_capacity", "rooms_count",
    "location_latitude", "location_longitude",]
df = pd.read_parquet(file_path, columns=columns)

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# همبستگی اسپیرمن
corr = df[columns].corr(method="spearman")

# اسم ستون‌ها فارسی
persian_labels = {
    "price_value": "مبلغ قیمت",
    "land_size": "متراژ زمین",
    "building_size": "زیربنا",
    "regular_person_capacity": "ظرفیت نفرات",
    "rooms_count": "تعداد اتاق‌ها",
    "location_latitude": "عرض جغرافیایی",
    "location_longitude": "طول جغرافیایی",
}
labels = [fix_fa(persian_labels[c]) for c in columns]

# اعداد رو نمودار
annot = corr.map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")

# رنگ سبز و صورتی
cmap = mcolors.LinearSegmentedColormap.from_list(
    "pink_green", ["#FF69B4", "#FFFFFF", "#2E8B57"])
cmap.set_bad("#BDBDBDC7")

# رسم نمودار
fig, ax = plt.subplots(figsize=(14, 12))
fig.subplots_adjust(top=0.90, bottom=0.20, left=0.20, right=0.88)

sns.heatmap(
    corr, annot=annot, fmt="", cmap=cmap,
    vmin=-1, vmax=1, center=0, square=True,
    linewidths=1.5, linecolor="white",
    xticklabels=labels, yticklabels=labels,
    annot_kws={"fontsize": 12, "fontweight": "bold", "color": "black"},
    cbar_kws={"shrink": 0.80}, ax=ax,)

ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
ax.set_yticklabels(labels, rotation=0, fontsize=12)
ax.tick_params(axis="x", pad=10)
ax.tick_params(axis="y", pad=10)

ax.set_title(fix_fa("ماتریس همبستگی متغیرها"), fontsize=18, fontweight="bold", pad=20)

cbar = ax.collections[0].colorbar
cbar.set_label(fix_fa("ضریب همبستگی"), fontsize=12, rotation=270, labelpad=20)

plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()