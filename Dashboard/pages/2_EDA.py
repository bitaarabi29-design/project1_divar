import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
import jdatetime
from data_utils import load_data

st.set_page_config(page_title="EDA", layout="wide")
st.title("📈 EDA & Descriptive Statistics")

# ---------- Persian text helper (from your scripts) ----------
plt.rcParams['font.family'] = 'Tahoma'

def fix_fa(text):
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)

# ---------- Load cleaned data (the output of Final_Preprocess.ipynb) ----------
df = load_data()
df['created_at_month'] = pd.to_datetime(df['created_at_month'], errors='coerce')

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("🎛️ Filters")
    cities = st.multiselect("City", sorted(df['city_slug'].dropna().unique()))
    deals  = st.multiselect(
        "Deal type",
        sorted(df['deal_type'].dropna().unique()),
        default=list(df['deal_type'].dropna().unique()))
    resid  = st.radio("Residential only?",
                      ["All", "Residential", "Non-residential"])

f = df.copy()
if cities: f = f[f['city_slug'].isin(cities)]
if deals:  f = f[f['deal_type'].isin(deals)]
if resid == "Residential":     f = f[f['is_residential_unit']]
if resid == "Non-residential": f = f[~f['is_residential_unit']]
st.caption(f"**{len(f):,}** ads match the current filters.")

tabs = st.tabs([f"Descriptive Statistics #{i}" for i in range(1, 10)])

# ==================================================================
# Q1 — cat2_slug and cat3_slug distributions
# ==================================================================
with tabs[0]:
    st.markdown("**Q1. توزیع آگهی‌های موجود در دسته‌های مختلف را برای دسته‌بندی سطح دو و سطح سه رسم کنید.**")
    st.plotly_chart(
        px.histogram(f, x="cat2_slug", title="cat2_slug"),
        use_container_width=True)
    st.plotly_chart(
        px.histogram(f, x="cat3_slug", title="cat3_slug"),
        use_container_width=True)

# ==================================================================
# Q2 — construction_year histogram
# ==================================================================
with tabs[1]:
    st.markdown("**Q2. هیستوگرام سال ساخت را رسم کنید.**")
    st.plotly_chart(
        px.histogram(f, x="construction_year", title="construction_year"),
        use_container_width=True)

# ==================================================================
# Q3 — ads per month for residential sell/rent
# ==================================================================
with tabs[2]:
    st.markdown("**Q3. تعداد آگهی‌های منتشر شده در ماه‌های مختلف را برای فروش و اجاره بررسی کنید.**")
    sell_rent = f[f["cat2_slug"].isin(["residential-sell", "residential-rent"])].copy()
    sell_rent["created_at_month"] = pd.to_datetime(sell_rent["created_at_month"])
    sell_rent["month"] = sell_rent["created_at_month"].dt.month
    sell_rent["month"] = sell_rent["month"].map({
        1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June",
        7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"})
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    st.plotly_chart(
        px.histogram(sell_rent, x="month", color="cat2_slug",
                     barmode="group",
                     category_orders={"month": month_order},
                     title="Ads Per Month"),
        use_container_width=True)

# ==================================================================
# Q4 — price_value by cat3_slug
# ==================================================================
with tabs[3]:
    st.markdown("**Q4. توزیع قیمت فروش (price_value) را برای دسته‌بندی‌های سطح سه رسم کنید.**")
    sell_prices = f[f["price_value"].notna()]
    st.plotly_chart(
        px.histogram(sell_prices, x="price_value", color="cat3_slug",
                     nbins=40, title="price_value"),
        use_container_width=True)

# ==================================================================
# Q5 — geographic heatmap
# ==================================================================
with tabs[4]:
    st.markdown("""
    **Q5. بر روی نقشه‌ی جغرافیایی heatmap آگهی‌های هر منطقه را رسم کنید.
    تراکم آگهی‌ها کدام منطقه بیشتر است؟**
    """)

    map_data = f.dropna(subset=["location_latitude", "location_longitude"])
    if len(map_data) > 20000:
        map_data = map_data.sample(20000, random_state=1)

    try:
        fig = px.density_map(
            map_data,
            lat='location_latitude',
            lon='location_longitude',
            radius=5,
            zoom=5,
            map_style='open-street-map',
            title='Ad density',
        )
        st.plotly_chart(fig, use_container_width=True)
    except AttributeError:
        # Fallback: Streamlit's built-in map (always works, no plotly needed)
        st.map(map_data.rename(columns={
            'location_latitude': 'lat',
            'location_longitude': 'lon',
        })[['lat', 'lon']], zoom=5)

    st.markdown("**Top 5 neighborhoods by ad count:**")
    st.write(f["neighborhood_slug"].value_counts().head(5))

# ==================================================================
# Q6 — rent trend by Jalali month
# ==================================================================
with tabs[5]:
    st.markdown("""
    **Q6. ترند میانگین قیمت اجاره بر حسب ماه‌های قرار گرفتن آگهی‌ها رسم کنید.
    (دقت کنید که ماه‌ها باید به تاریخ شمسی و خوانا باشند.)**
    """)

    df_rent = f[f['deal_type'] == 'rent'].copy()
    df_rent['date_dt'] = pd.to_datetime(df_rent['created_at_month'], errors='coerce')
    df_rent = df_rent[
        df_rent['rent_value'].notna()
        & (df_rent['rent_value'] >= 10000)
        & (df_rent['rent_value'] <= 500000000)
    ]

    if len(df_rent) == 0:
        st.warning("No rent ads match the current filters.")
    else:
        trend_df = (
            df_rent
            .groupby('date_dt')['rent_value']
            .median()
            .reset_index()
            .sort_values('date_dt')
        )

        def to_jalali_month(dt):
            j_date = jdatetime.date.fromgregorian(date=dt.date())
            months_fa = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر',
                         'مرداد', 'شهریور', 'مهر', 'آبان',
                         'آذر', 'دی', 'بهمن', 'اسفند']
            return fix_fa(f'{months_fa[j_date.month - 1]} {j_date.year}')

        trend_df['jalali_month'] = trend_df['date_dt'].apply(to_jalali_month)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(trend_df['jalali_month'], trend_df['rent_value'],
                marker='o', linestyle='-', color='#761314', linewidth=2.5)
        ax.set_title(fix_fa('ترند میانه قیمت اجاره بر حسب ماه‌های شمسی'),
                     fontsize=15, pad=15)
        ax.set_xlabel(fix_fa('ماه ثبت آگهی'), fontsize=12)
        ax.set_ylabel(fix_fa('میانه قیمت اجاره (تومان)'), fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==================================================================
# Q7 — real price trend 1400–1403
# ==================================================================
with tabs[6]:
    st.markdown("""
    **Q7. در طول زمان قیمت‌های اسمی افزایش پیدا می‌کنند اما این افزایش لزوما به معنی
    بالارفتن ارزش واقعی ملک نیست و می‌تواند ناشی از تورم باشد. به ازای میانگین مبلغ
    قیمت (price_value) در سال‌های ۱۴۰۰ تا ۱۴۰۳ قیمت حقیقی را محاسبه کنید و بررسی
    کنید ترند قیمت حقیقی چگونه است.**
    """)

    def gregorian_to_jalali_year(date):
        if pd.isna(date):
            return None
        year = date.year
        if year == 2024:
            start = pd.Timestamp('2024-03-20')
        else:
            start = pd.Timestamp(f'{year}-03-21')
        return year - 621 if date >= start else year - 622

    d7 = f.copy()
    d7['date_dt'] = pd.to_datetime(d7['created_at_month'], errors='coerce')
    d7['year'] = d7['date_dt'].apply(gregorian_to_jalali_year)
    d7 = d7[d7['year'].isin([1400, 1401, 1402, 1403])]

    if len(d7) == 0:
        st.warning("No data from years 1400–1403 matches the current filters.")
    else:
        inflation = {1400: 46.2, 1401: 53.1, 1402: 47.4, 1403: 35.8}
        price_index = {1400: 100}
        for year in [1401, 1402, 1403]:
            price_index[year] = price_index[year - 1] * (1 + inflation[year] / 100)

        annual_price = (
            d7.groupby('year')['price_value'].mean().reset_index()
        )
        annual_price['price_index'] = annual_price['year'].map(price_index)
        base_year = 1403
        annual_price['real_price'] = (
            annual_price['price_value']
            * price_index[base_year]
            / annual_price['price_index']
        )

        years = annual_price['year'].tolist()
        year_labels = [fix_fa(f'سال {int(y)}') for y in years]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(years, annual_price['real_price'],
                color='#4D0547', marker='o', markersize=8, linewidth=3.5)
        ax.set_xticks(years)
        ax.set_xticklabels(year_labels, fontsize=11)

        def format_y(value, position):
            return fix_fa(f'{value:,.0f}')

        ax.yaxis.set_major_formatter(FuncFormatter(format_y))
        ax.set_xlabel(fix_fa('سال'), fontsize=13)
        ax.set_ylabel(fix_fa('قیمت حقیقی'), fontsize=13)
        ax.set_title(fix_fa('روند قیمت حقیقی ملک در سال‌های ۱۴۰۰ تا ۱۴۰۳'),
                     fontsize=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.8, linewidth=1.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==================================================================
# Q8 — correlation heatmap
# ==================================================================
with tabs[7]:
    st.markdown("""
    **Q8. ماتریس هم‌بستگی را برای مبلغ قیمت، متراژ زمین، زیربنا، ظرفیت نفرات،
    تعداد اتاق‌ها و طول و عرض جغرافیایی رسم نمایید.**
    """)

    columns = ["price_value", "land_size", "building_size",
               "regular_person_capacity", "rooms_count",
               "location_latitude", "location_longitude"]

    # keep only columns that actually exist in the parquet
    columns = [c for c in columns if c in f.columns]
    d8 = f[columns].apply(pd.to_numeric, errors="coerce")
    correlation_matrix = d8.corr(method="pearson")

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
    annot_matrix = correlation_matrix.map(
        lambda x: "N/A" if pd.isna(x) else f"{x:.2f}"
    )

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("#BDBDBDC7")

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(top=0.92, bottom=0.22, left=0.22, right=0.90)
    sns.heatmap(
        correlation_matrix,
        annot=annot_matrix, fmt="",
        cmap=cmap, vmin=-1, vmax=1, center=0, square=True,
        linewidths=1.5, linecolor="white",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"fontsize": 11, "fontweight": "bold", "color": "black"},
        cbar_kws={"shrink": 0.80}, ax=ax,
    )
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)
    ax.set_title(fix_fa("ماتریس همبستگی متغیرها"),
                 fontsize=16, fontweight="bold", pad=18)
    cbar = ax.collections[0].colorbar
    cbar.set_label(fix_fa("ضریب همبستگی"),
                   fontsize=11, rotation=270, labelpad=18)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ==================================================================
# Q9 — amenities by Tehran neighborhood
# ==================================================================
with tabs[8]:
    st.markdown("""
    **Q9. می‌خواهیم بررسی کنیم خانه‌هایی که دارای بالکن، آسانسور، نگهبان،
    باربیکیو و استخر هستند عمدتا در کدام مناطق قرار دارند. با نمودار مناسب
    این موضوع را نشان دهید.**
    """)

    city_clean = f["city_slug"].astype(str).str.strip().str.lower()
    df_tehran = f[
        city_clean.str.contains("tehran", na=False) | city_clean.eq("تهران")
    ].copy()

    neighborhood_fa = {
        "saadat_abad": "سعادت‌آباد", "saadatabad": "سعادت‌آباد",
        "janat_abad": "جنت‌آباد", "janatabad": "جنت‌آباد",
        "elahiyeh": "الهیه", "elahieh": "الهیه",
        "simon_bolivar": "سیمون بولیوار", "simon-bolivar": "سیمون بولیوار",
        "shahrak_gharb": "شهرک غرب", "shahrak-gharb": "شهرک غرب",
        "punak": "پونک", "punak_shomali": "پونک شمالی", "punak_jonoobi": "پونک جنوبی",
        "farahzad": "فرحزاد", "marzdaran": "مرزداران", "bagh_feyz": "باغ فیض",
        "shahran": "شهران", "ferdows": "فردوس", "ekbatan": "اکباتان",
        "ariya_shahr": "آریاشهر", "yousef_abad": "یوسف‌آباد", "yusef_abad": "یوسف‌آباد",
        "mirdamad": "میرداماد", "vanak": "ونک", "qolhak": "قلهک", "gholhak": "قلهک",
        "farmanieh": "فرمانیه", "kamranieh": "کامرانیه", "kamraniyeh": "کامرانیه",
        "zafaraniyeh": "زعفرانیه", "zaferaniyeh": "زعفرانیه", "niavaran": "نیاوران",
        "tajrish": "تجریش", "velenjak": "ولنجک", "darakeh": "درکه",
        "qeytarieh": "قیطریه", "qeytarie": "قیطریه", "pasdaran": "پاسداران",
        "tehranpars": "تهرانپارس", "narmak": "نارمک", "lavisan": "لویزان",
        "hakimiyeh": "حکیمیه", "shamsabad": "شمس‌آباد", "sadeghieh": "صادقیه",
        "sadeghiyeh": "صادقیه", "gisha": "گیشا", "kuy_nasr": "کوی نصر",
        "moniriyeh": "منیریه", "jomhouri": "جمهوری", "yaftabad": "یافت‌آباد",
        "chitgar": "چیتگر", "shahrak_azadi": "شهرک آزادی", "shahrak_baqeri": "شهرک باقری",
        "aghdasiyeh": "اقدسیه", "mahmodiyeh": "محمودیه", "mahmoodieh": "محمودیه",
        "fereshteh": "فرشته", "jordaan": "جردن", "jordan": "جردن",
        "darrous": "دروس", "darus": "دروس", "ekhtiyariyeh": "اختیاریه",
        "niayesh": "نیایش", "zargandeh": "زرگنده", "abbas_abad": "عباس‌آباد",
        "seyed_khandan": "سیدخندان", "motahari": "مطهری", "fatemi": "فاطمی",
        "amiriyeh": "امیریه", "nazem_abad": "نظام‌آباد", "nizam_abad": "نظام‌آباد",
        "tehran_no": "تهران‌نو", "tehran_nou": "تهران‌نو", "tehran_pars": "تهرانپارس",
        "nazi_abad": "نازی‌آباد", "joobsh": "جوب‌شیر",
    }

    df_tehran["neighborhood_clean"] = (
        df_tehran["neighborhood_slug"].astype(str).str.strip().str.lower()
        .str.replace("-", "_", regex=False)
    )
    df_tehran["neighborhood_fa"] = df_tehran["neighborhood_clean"].map(neighborhood_fa)

    features = {
        "has_balcony": {"title": "درصد خانه‌های دارای بالکن در محله‌های تهران", "color": "#93C078"},
        "has_elevator": {"title": "درصد خانه‌های دارای آسانسور در محله‌های تهران", "color": "#EC6DA2"},
        "has_security_guard_combined": {"title": "درصد خانه‌های دارای نگهبانی/سرایداری در محله‌های تهران", "color": "#B5A5DA"},
        "has_barbecue_combined": {"title": "درصد خانه‌های دارای باربیکیو در محله‌های تهران", "color": "#E8BC5A"},
        "has_pool_combined": {"title": "درصد خانه‌های دارای استخر در محله‌های تهران", "color": "#7CDBBE"},
    }

    def convert_feature(series):
        true_values = {True, 1, "1", "1.0", "true", "True", "TRUE", "yes", "Yes", "YES", "دارد", "دارد "}
        false_values = {False, 0, "0", "0.0", "false", "False", "FALSE", "no", "No", "NO", "ندارد", "ندارد "}
        result = pd.Series([None] * len(series), index=series.index, dtype="object")
        result[series.isin(true_values)] = True
        result[series.isin(false_values)] = False
        return result

    rendered_any = False
    for feature, info in features.items():
        if feature not in df_tehran.columns:
            st.info(f"Column `{feature}` not found in data — skipping.")
            continue

        temp = df_tehran[["neighborhood_fa", feature]].copy()
        temp[feature] = convert_feature(temp[feature])
        temp = temp.dropna(subset=["neighborhood_fa"])
        temp_known = temp.dropna(subset=[feature])

        if len(temp_known) == 0:
            st.info(f"No data for `{feature}` — skipping.")
            continue

        # cast to numeric so sum works reliably
        temp_known[feature] = temp_known[feature].astype(int)

        result = (
            temp_known.groupby("neighborhood_fa")[feature]
            .agg(total_known="count", has_feature="sum")
            .reset_index()
        )
        result["percentage"] = result["has_feature"] / result["total_known"] * 100
        result = result.sort_values("percentage", ascending=True)

        fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(result))))
        ax.barh(result["neighborhood_fa"].apply(fix_fa),
                result["percentage"], color=info["color"])
        ax.set_title(fix_fa(info["title"]), fontsize=16, fontweight="bold", pad=12)
        ax.set_xlabel(fix_fa("درصد خانه‌ها (%)"), fontsize=12)
        ax.set_ylabel(fix_fa("محله"), fontsize=12)
        max_val = result["percentage"].max()
        ax.set_xlim(0, max(max_val * 1.15, 10))
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(ax.patches, result["percentage"]):
            ax.text(value + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        rendered_any = True

    if not rendered_any:
        st.warning("No amenity charts could be rendered for the current filters.")