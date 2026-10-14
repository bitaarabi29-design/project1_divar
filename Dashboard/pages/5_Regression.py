import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Price Prediction (Regression)",
    page_icon="📈",
    layout="wide"
)

# اعمال RTL صرفاً روی متن‌ها، پاراگراف‌ها، لیست‌ها و عناوین مارک‌داون
st.markdown("""
<style>
    /* اعمال راست به چپ فقط بر روی المان‌های متنی */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown blockquote {
        direction: rtl !important;
        text-align: right !important;
    }

    /* کدها و فرمول‌های ریاضی درون متن چپ به راست بمانند */
    .stMarkdown code,
    .stMarkdown pre,
    .stMarkdown .katex {
        direction: ltr !important;
        unicode-bidi: embed;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 پیش‌بینی قیمت املاک (Regression)")

ASSETS_DIR = Path(__file__).parent.parent / "assets" / "regression_plots"

# Helper to render images safely without app crash
def render_plot(filename: str, caption: str = ""):
    img_path = ASSETS_DIR / filename
    if img_path.exists():
        st.image(str(img_path), caption=caption, use_container_width=True)
    else:
        st.info(f"نمودار پیش‌رندر شده یافت نشد: `{filename}`. تصویر را در مسیر `assets/regression_plots/` قرار دهید.")

# ---------------------------------------------------------
# Tabs Definition
# ---------------------------------------------------------
tab_overview, tab_linear, tab_tree_exp, tab_val_comp, tab_test_eval, tab_errors = st.tabs([
    "نمای کلی داده و توزیع",
    "مدل‌های خطی و مهندسی ویژگی",
    "آزمایش‌های مدل‌های درختی",
    "مقایسه مدل‌ها در Validation",
    "ارزیابی نهایی روی Test",
    "تحلیل خطا و اهمیت ویژگی‌ها"
])

# ---------------------------------------------------------
# Tab 1: Overview & Data Prep
# ---------------------------------------------------------
with tab_overview:
    st.markdown("""
### هدف و آماده‌سازی داده

ستون‌های متنی `title` و `description` به دلیل ساخته شدن فیچرهای `mentions_*` کنار گذاشته شده‌اند. آگهی‌ها فقط به نوع معامله فروش و اجاره محدود شده و رکوردهای فاقد متراژ، ناقص یا آگهی‌های نیازمندی حذف شدند تا در نهایت **۸۵۷,۰۰۹** آگهی به عنوان جامعه اصلی باقی بماند.

- **توزیع معامله:** ۵۲۷,۲۸۴ فروش و ۳۲۹,۷۲۵ اجاره
- **تقسیم داده:** تفکیک طبقه‌ای (Stratified Split بر روی `deal_type`) با نسبت ۸۰٪ Train و ۱۰٪ Validation و ۱۰٪ Test
- **هدف مدل:** لگاریتم قیمت واقعی (`log(unified_price)`) به دلیل چولگی شدید قیمت خام.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### چولگی قیمت و اثر تبدیل لگاریتمی")
        render_plot("5_skewness_hist.png", "هیستوگرام توزیع قیمت و تبدیل log")
    with col2:
        st.markdown("#### رابطه لگاریتم قیمت در برابر متراژ")
        render_plot("4_hexbin_price_size.png", "نمودار Hexbin قیمت بر حسب متراژ در دسته‌های مختلف")

    st.markdown("---")
    st.markdown("#### وضعیت داده‌های مفقود و Outlierها")
    c1, c2 = st.columns(2)
    with c1:
        render_plot("6_missing_values_stacked_bar.png", "بررسی مقادیر گمشده به تفکیک ویژگی‌ها")
    with c2:
        render_plot("7_iqr_outliers_boxplot.png", "شناسایی Outlierها بر اساس IQR در log(price_per_m2)")

# ---------------------------------------------------------
# Tab 2: Linear Models & PCA
# ---------------------------------------------------------
with tab_linear:
    st.markdown("""
### مدل‌های خطی، چندجمله‌ای و کاهش بعد با PCA

در ابتدا مدل خطی Baseline با ایمپیوتیشن میانه و One-Hot Encoding پیاده‌سازی شد. سپس برای مدیریت بعد بالای فیچرهای مکانی (شهر و محله) از `TargetEncoder`، ایجاد ترم‌های درجه ۲ (`PolynomialFeatures`) و فشرده‌سازی با `PCA` استفاده شد.
    """)

    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        st.markdown("#### واریانس تجمعی مؤلفه‌های PCA")
        render_plot("3_cumulative_variance_pca.png", "Cumulative Explained Variance در مؤلفه‌های PCA")

    with col_l2:
        st.markdown("#### نتایج اعمال PCA روی مدل خطی")
        st.markdown("""
- بیشترین دقت در مدل‌های خطی با **۶۰ مؤلفه اول PCA** به دست آمد.
- مقدار $R^2$ روی log قیمت برای فروش به **۰.۵۹۲** و برای اجاره به **۰.۷۱۵** رسید.
- با این وجود، مدل‌های خطی توانایی یادگیری الگوهای غیرخطی پیچیده قیمت را به طور کامل ندارند.
        """)
        pca_metrics = pd.DataFrame({
            "نوع معامله": ["فروش (Sale)", "اجاره (Rent)"],
            "R² (log)": [0.592, 0.715],
            "MAE (میلیون تومان)": [3408.27, 428.80],
            "مدل": ["Polynomial + PCA (k=60)", "Polynomial + PCA (k=60)"]
        })
        st.dataframe(pca_metrics, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: Tree Models Experiments
# ---------------------------------------------------------
with tab_tree_exp:
    st.markdown("""
### اثر Outliers و داده‌های مفقود بر مدل‌های درختی

اثر حذف Outlierها و شیوه مدیریت Missing Values بر روی الگوریتم‌های مبتنی بر درخت (LightGBM) سنجیده شد.
    """)

    st.markdown("#### ۱. اثر حذف داده‌های پرت (Outliers)")
    outlier_df = pd.DataFrame({
        "معامله / داده آموزش": [
            "فروش، بدون داده‌های پرت (Outliers removed)",
            "فروش، کل داده‌های آموزش (All train)",
            "اجاره، بدون داده‌های پرت (Outliers removed)",
            "اجاره، کل داده‌های آموزش (All train)"
        ],
        "R² (log)": [0.723, 0.731, 0.769, 0.785],
        "R² (تومانی)": [0.214, 0.494, 0.371, 0.395],
        "MAE (میلیون تومان)": [2564.58, 2444.57, 332.97, 310.68]
    })
    st.dataframe(outlier_df, hide_index=True, use_container_width=True)

    st.markdown("""
> **یافته نوت‌بوک:** برای LightGBM حذف outlier زیان‌آور است. با آموزش روی کل داده Train، مقدار $R^2$ تومانی فروش از ۰.۲۱ به ۰.۴۹ و اجاره از ۰.۳۷ به ۰.۳۹ افزایش می‌یابد؛ بنابراین مدل‌های درختی روی **کل داده آموزش** یادگیری را انجام می‌دهند.
    """)

    st.markdown("#### ۲. شیوه برخورد با Missing Values")
    missing_df = pd.DataFrame({
        "معامله / نحوه برخورد": [
            "فروش، حفظ مقادیر مفقود (Missing kept)",
            "فروش، پر کردن با میانه و صفر (Missing filled)",
            "اجاره، حفظ مقادیر مفقود (Missing kept)",
            "اجاره، پر کردن با میانه و صفر (Missing filled)"
        ],
        "R² (log)": [0.731, 0.731, 0.785, 0.785],
        "R² (تومانی)": [0.494, 0.420, 0.395, 0.400],
        "MAE (میلیون تومان)": [2444.57, 2482.98, 310.68, 307.75]
    })
    st.dataframe(missing_df, hide_index=True, use_container_width=True)

    st.markdown("""
> **یافته نوت‌بوک:** پر کردن دستی مقادیر مفقود مزیتی ایجاد نمی‌کند و در فروش خطای MAE را بدتر می‌کند؛ بنابراین درخت‌ها به صورت نیتیو با Missing Values کار می‌کنند.
    """)

# ---------------------------------------------------------
# Tab 4: Validation Comparison
# ---------------------------------------------------------
with tab_val_comp:
    st.markdown("""
### مقایسه عملکرد مدل‌ها روی داده Validation

پیش‌بینی نهایی از ترکیب و میانگین‌گیری انسمبل LightGBM و CatBoost به دست آمده است.
    """)

    val_summary = pd.DataFrame({
        "نوع معامله": ["فروش", "فروش", "فروش", "فروش", "اجاره", "اجاره", "اجاره", "اجاره"],
        "مدل": [
            "Linear (Poly + PCA)",
            "LightGBM",
            "CatBoost",
            "Ensemble (LGBM + CatBoost)",
            "Linear (Poly + PCA)",
            "LightGBM",
            "CatBoost",
            "Ensemble (LGBM + CatBoost)"
        ],
        "R² (log)": [0.592, 0.737, 0.729, 0.739, 0.715, 0.789, 0.785, 0.791],
        "R² (تومانی)": [0.087, 0.467, 0.399, 0.449, -0.441, 0.413, 0.500, 0.455],
        "MAE (میلیون تومان)": [3408.27, 2416.49, 2565.34, 2432.34, 428.80, 305.54, 312.76, 303.77]
    })
    st.dataframe(val_summary, hide_index=True, use_container_width=True)

    st.markdown("""
- مدل‌های درختی ارتقای چشمگیری نسبت به مدل‌های خطی دارند.
- میانگین پیش‌بینی دو مدل **LightGBM + CatBoost** در هر دو گروه بیشترین دقت $R^2$ روی لگاریتم را ثبت کرد و به عنوان مدل نهایی برگزیده شد.
    """)

# ---------------------------------------------------------
# Tab 5: Final Evaluation on Test
# ---------------------------------------------------------
with tab_test_eval:
    st.markdown("""
### ارزیابی نهایی روی مجموعه داده Test

مدل‌ها تنها یک بار روی ۱۰٪ پایانی تفکیک‌شده (Test) آزموده شدند.
    """)

    test_df = pd.DataFrame({
        "معامله": ["فروش", "فروش", "فروش", "اجاره", "اجاره", "اجاره"],
        "مدل": [
            "Baseline (خطی عمومی)",
            "Linear (Poly + PCA)",
            "Ensemble (LightGBM + CatBoost)",
            "Baseline (خطی عمومی)",
            "Linear (Poly + PCA)",
            "Ensemble (LightGBM + CatBoost)"
        ],
        "R² (log)": [0.549, 0.590, 0.734, 0.685, 0.715, 0.798],
        "R² (تومانی)": [0.218, -0.486, 0.469, 0.071, -0.042, 0.205],
        "MAE (میلیون تومان)": [3600.81, 3472.79, 2471.83, 479.83, 445.85, 322.74]
    })
    st.dataframe(test_df, hide_index=True, use_container_width=True)

    st.markdown("""
### اثر آگهی‌های شدید (Extreme Ads) در Test
همان کران‌های IQR به دست آمده از داده Train روی Test اعمال شد تا مشخص شود حساسیت معیارها به آگهی‌های با قیمت غیرعادی چقدر است:
    """)

    extreme_df = pd.DataFrame({
        "معامله": ["فروش", "فروش (داخل مرز IQR)", "اجاره", "اجاره (داخل مرز IQR)"],
        "سهم داده": ["۱۰۰٪", "۹۸.۰٪", "۱۰۰٪", "۹۷.۶٪"],
        "R² (log)": [0.734, 0.758, 0.798, 0.858],
        "R² (تومانی)": [0.469, 0.506, 0.205, 0.699],
        "MAE (میلیون تومان)": [2471.83, 2183.03, 322.74, 240.05]
    })
    st.dataframe(extreme_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# Tab 6: Error Analysis & Feature Importance
# ---------------------------------------------------------
with tab_errors:
    st.markdown("""
### تحلیل خطای نسبی به تفکیک نوع ملک و اهمیت ویژگی‌ها

خطای نسبی هر آگهی با فرمول `|Predicted / Actual - 1|` محاسبه شده است.
    """)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### پراکندگی پیش‌بینی در برابر قیمت واقعی (Hexbin)")
        render_plot("1_hexbin_pred_vs_actual.png", "مقایسه لگاریتم قیمت واقعی در برابر پیش‌بینی مدل")

    with col_e2:
        st.markdown("#### اهمیت متغیرها در مدل LightGBM")
        render_plot("2_feature_importance.png", "سهم متغیرهای برتر از کل Gain مدل")

    st.markdown("#### میانه خطای نسبی در دسته‌بندی‌های مختلف")
    error_cat = pd.DataFrame({
        "دسته‌بندی (cat3_slug)": [
            "apartment-sell (آپارتمان فروش)",
            "apartment-rent (آپارتمان اجاره)",
            "house-villa-rent (ویلا اجاره)",
            "office-rent (اداری اجاره)",
            "office-sell (اداری فروش)",
            "house-villa-sell (ویلا فروش)",
            "shop-rent (مغازه اجاره)",
            "industry-agriculture-business-rent (صنعتی/کشاورزی اجاره)",
            "shop-sell (مغازه فروش)",
            "plot-old (کلنگی و زمین)",
            "industry-agriculture-business-sell (صنعتی/کشاورزی فروش)"
        ],
        "تعداد آگهی": [28743, 20189, 6009, 1888, 393, 10828, 4157, 730, 1675, 10347, 742],
        "میانه خطای نسبی": ["11.5%", "12.0%", "20.0%", "20.5%", "22.9%", "27.0%", "41.3%", "42.4%", "43.6%", "50.2%", "61.0%"]
    })
    st.dataframe(error_cat, hide_index=True, use_container_width=True)

    st.markdown("""
### نتیجه‌گیری نهایی
- مدل تلفیقی روی دادگان آزمون (Test)، شاخص $R^2$ روی لگاریتم قیمت را برای فروش از **۰.۵۵ به ۰.۷۳** و برای اجاره از **۰.۶۹ به ۰.۸۰** ارتقا داده و MAE را حدود **۳۰ درصد کاهش** داده است.
- خطای پیش‌بینی برای آپارتمان‌ها در حد عالی (حدود ۱۲ درصد) است اما برای زمین و املاک صنعتی/کشاورزی خطای بیشتری (۴۰ الی ۶۰ درصد) ثبت می‌شود.
- تعیین‌کننده‌ترین ویژگی‌ها شامل **شهر، متراژ، محله و ویژگی مکانی قیمت همسایه‌ها (`neighbor_price`)** می‌باشند.
    """)
