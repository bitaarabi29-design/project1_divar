# وارد کردن کتابخانه‌ها 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# از راست به چپ کردن متن فارسی
def fix_fa(text):
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)

# خواندن دیتای مورد نیاز
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "final_data.csv")
df = pd.read_csv(desktop_path, low_memory=False)

city_clean = (
    df["city_slug"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_tehran = df[
    city_clean.str.contains("tehran", na=False) |
    city_clean.eq("تهران")
].copy()

# نام فارسی محله‌ها
neighborhood_fa = {
    "saadat_abad": "سعادت‌آباد",
    "saadatabad": "سعادت‌آباد",
    "janat_abad": "جنت‌آباد",
    "janatabad": "جنت‌آباد",
    "elahiyeh": "الهیه",
    "elahieh": "الهیه",
    "simon_bolivar": "سیمون بولیوار",
    "simon-bolivar": "سیمون بولیوار",
    "shahrak_gharb": "شهرک غرب",
    "shahrak-gharb": "شهرک غرب",
    "punak": "پونک",
    "punak_shomali": "پونک شمالی",
    "punak_jonoobi": "پونک جنوبی",
    "farahzad": "فرحزاد",
    "marzdaran": "مرزداران",
    "bagh_feyz": "باغ فیض",
    "shahran": "شهران",
    "ferdows": "فردوس",
    "ekbatan": "اکباتان",
    "ariya_shahr": "آریاشهر",
    "yousef_abad": "یوسف‌آباد",
    "yusef_abad": "یوسف‌آباد",
    "mirdamad": "میرداماد",
    "vanak": "ونک",
    "qolhak": "قلهک",
    "gholhak": "قلهک",
    "farmanieh": "فرمانیه",
    "kamranieh": "کامرانیه",
    "kamraniyeh": "کامرانیه",
    "zafaraniyeh": "زعفرانیه",
    "zaferaniyeh": "زعفرانیه",
    "niavaran": "نیاوران",
    "tajrish": "تجریش",
    "velenjak": "ولنجک",
    "darakeh": "درکه",
    "qeytarieh": "قیطریه",
    "qeytarie": "قیطریه",
    "pasdaran": "پاسداران",
    "tehranpars": "تهرانپارس",
    "narmak": "نارمک",
    "lavisan": "لویزان",
    "hakimiyeh": "حکیمیه",
    "shamsabad": "شمس‌آباد",
    "sadeghieh": "صادقیه",
    "sadeghiyeh": "صادقیه",
    "gisha": "گیشا",
    "kuy_nasr": "کوی نصر",
    "moniriyeh": "منیریه",
    "jomhouri": "جمهوری",
    "yaftabad": "یافت‌آباد",
    "chitgar": "چیتگر",
    "shahrak_azadi": "شهرک آزادی",
    "shahrak_baqeri": "شهرک باقری",
    "aghdasiyeh": "اقدسیه",
    "mahmodiyeh": "محمودیه",
    "mahmoodieh": "محمودیه",
    "fereshteh": "فرشته",
    "jordaan": "جردن",
    "jordan": "جردن",
    "darrous": "دروس",
    "darus": "دروس",
    "ekhtiyariyeh": "اختیاریه",
    "niayesh": "نیایش",
    "zargandeh": "زرگنده",
    "abbas_abad": "عباس‌آباد",
    "seyed_khandan": "سیدخندان",
    "motahari": "مطهری",
    "fatemi": "فاطمی",
    "amiriyeh": "امیریه",
    "nazem_abad": "نظام‌آباد",
    "nizam_abad": "نظام‌آباد",
    "tehran_no": "تهران‌نو",
    "tehran_nou": "تهران‌نو",
    "tehran_pars": "تهرانپارس",
    "nazi_abad": "نازی‌آباد",
    "joobsh": "جوب‌شیر",
}

# تمیزکاری نام محله‌ها
df_tehran["neighborhood_clean"] = (
    df_tehran["neighborhood_slug"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace("-", "_", regex=False)
)

# مپ کردن نام‌های فارسی محله‌ها
df_tehran["neighborhood_fa"] = df_tehran["neighborhood_clean"].map(neighborhood_fa)

# تغییر در فیچرها: به‌روزرسانی اسامی ستون‌های جدید استخراج‌شده
features = {
    "has_balcony": {
        "title": "درصد خانه‌های دارای بالکن در محله‌های تهران",
        "color": "#93C078"
    },
    "has_elevator": {
        "title": "درصد خانه‌های دارای آسانسور در محله‌های تهران",
        "color": "#EC6DA2"
    },
    "is_security_guard_house_tehran": {  # ستون جدید استخراج شده برای نگهبانی/سرایداری
        "title": "درصد خانه‌های دارای نگهبانی/سرایداری در محله‌های تهران",
        "color": "#B5A5DA"
    },
    "is_barbecue_house_tehran": {  # ستون جدید استخراج شده برای باربیکیو
        "title": "درصد خانه‌های دارای باربیکیو در محله‌های تهران",
        "color": "#E8BC5A"
    },
    "is_pool_house_tehran": {  # ستون جدید استخراج شده برای استخر
        "title": "درصد خانه‌های دارای استخر در محله‌های تهران",
        "color": "#7CDBBE"
    }
}

# تابع تبدیل مقادیر متنی و متغیرها به Boolean
def convert_feature(series):
    true_values = {
        True, 1, "1", "1.0",
        "true", "True", "TRUE",
        "yes", "Yes", "YES",
        "دارد", "دارد "
    }

    false_values = {
        False, 0, "0", "0.0",
        "false", "False", "FALSE",
        "no", "No", "NO",
        "ندارد", "ندارد "
    }

    result = pd.Series(np.nan, index=series.index)
    result[series.isin(true_values)] = True
    result[series.isin(false_values)] = False
    return result

# رسم نمودارها
for feature, info in features.items():
    if feature not in df_tehran.columns:
        continue

    temp = df_tehran[["neighborhood_fa", feature]].copy()
    temp[feature] = convert_feature(temp[feature])

    temp = temp.dropna(subset=["neighborhood_fa"])
    temp_known = temp.dropna(subset=[feature])

    result = (
        temp_known
        .groupby("neighborhood_fa")[feature]
        .agg(
            total_known="count",
            has_feature="sum"
        )
        .reset_index()
    )

    result["percentage"] = (
        result["has_feature"] /
        result["total_known"] * 100
    )

    result = result.sort_values("percentage", ascending=True)

    plt.figure(figsize=(14, 12))

    plt.barh(
        result["neighborhood_fa"].apply(fix_fa),
        result["percentage"],
        color=info["color"]
    )

    plt.title(
        fix_fa(info["title"]),
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        fix_fa("درصد خانه‌ها (%)"),
        fontsize=13
    )

    plt.ylabel(
        fix_fa("محله"),
        fontsize=13
    )

    max_val = result["percentage"].max() if len(result) > 0 else 100
    plt.xlim(0, max(max_val * 1.15, 10))

    plt.grid(
        axis="x",
        alpha=0.2
    )

    for bar, value in zip(plt.gca().patches, result["percentage"]):
        plt.text(
            value + (max_val * 0.01 if max_val > 0 else 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()