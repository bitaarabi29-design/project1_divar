import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
from scipy.stats import (mannwhitneyu, shapiro, ttest_ind, levene,
                         kruskal, chi2_contingency, gaussian_kde)
from statsmodels.stats.proportion import proportions_ztest
from data_utils import load_data

st.set_page_config(page_title="Hypothesis Tests", layout="wide")
st.title("🧪 Hypothesis Tests & Reclassification")

df = load_data()
df['created_at_month'] = pd.to_datetime(df['created_at_month'], errors='coerce')

tabs = st.tabs([
    "H1 — Metro vs small-city size",
    "H2 — Old vs new home size",
    "H3 — Business deed & sale price",
    "H4 — Luxury vs non-luxury amenities",
    "Binning — Apartment & Shop Size",
    "Reclassification — Climate, Season, Tourism",
    "Bonus #1 — Elevator & rebuild (Z-test)",
    "Bonus #2 — Building direction & price (Kruskal)",
    "Bonus #3 — Property & deed type (Chi²)",
])

# ==================================================================
# TAB 0 — H1: Metro vs small-city size (Mann-Whitney)
# ==================================================================
with tabs[0]:
    st.markdown("""
    **فرضیه:** با توجه به رشد مهاجرت به کلان‌شهرها و تراکم جمعیت، میانگین مساحت
    خانه‌های مسکونی در کلان‌شهرها نسبت به شهرهای کوچک کمتر است.
    """)

    st.markdown(
        """
        <div dir="rtl" align="right">
        <p>اگر p-value آزمون Shapiro-Wilk کمتر از ۰.۰۵ باشد، فرض نرمال بودن رد می‌شود.
        چون داده‌ها نرمال نیستند، از آزمون غیرپارامتری Mann-Whitney استفاده می‌کنیم.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div dir="rtl" align="right">
        <h3>فرضیه‌ی صفر و جایگزین</h3>
        <ul>
        <li><strong>H₀:</strong> میانگین مساحت کلان‌شهرها حداقل برابر شهرهای کوچک است
        (<code>μ_metro ≥ μ_small_city</code>).</li>
        <li><strong>H₁:</strong> میانگین مساحت کلان‌شهرها کمتر است
        (<code>μ_metro &lt; μ_small_city</code>).</li>
        </ul>
        <p>اگر <code>p &lt; 0.05</code> باشد، H₀ رد و H₁ پذیرفته می‌شود. آزمون یک‌طرفه است
        چون صورت مسئله جهت مشخصی داده («کمتر است»).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    classif_path = Path(__file__).parent.parent / "data" / "iran_city_classification.csv"

    if not classif_path.exists():
        st.warning(f"Reference file not found: `{classif_path.name}` — H1 cannot run.")
    else:
        @st.cache_data(show_spinner="Running H1 (Mann-Whitney) …")
        def _run_h1(df):
            city_classif = pd.read_csv(classif_path)
            city_classif.columns = ["name", "city_class"]
            city_classif["name"] = city_classif["name"].astype("string").str.lower()
            city_classif["city_class"] = city_classif["city_class"].replace({
                "کلان‌شهر": "metro",
                "شهر کوچک": "small_city",
            })
            city_map = city_classif.set_index("name")["city_class"].to_dict()

            data = df.copy()
            data["city_slug"] = data["city_slug"].astype("string").str.lower()
            data["city_class"] = data["city_slug"].map(city_map)
            data["city_class_imputed"] = data["city_class"].isna()
            data["city_class"] = data["city_class"].fillna("small_city")

            residential_cats = ["apartment-sell", "apartment-rent",
                                "house-villa-sell", "house-villa-rent",
                                "villa", "suite-apartment"]
            res = data[data["cat3_slug"].astype("string").isin(residential_cats)].copy()
            res["building_size"] = pd.to_numeric(res["building_size"], errors="coerce")

            col = "building_size"
            hard = res[col].notna() & ~res[col].between(20, 2500)
            res.loc[hard, col] = np.nan
            positive = res[col].where(res[col] > 0)
            log_vals = np.log10(positive)
            q1, q3 = log_vals.quantile(0.25), log_vals.quantile(0.75)
            iqr = q3 - q1
            low, high = 10 ** (q1 - 2.0 * iqr), 10 ** (q3 + 2.0 * iqr)
            soft = res[col].notna() & ~res[col].between(low, high)
            res.loc[soft, col] = np.nan

            metro_area = res.loc[res["city_class"] == "metro", col].dropna()
            small_area = res.loc[res["city_class"] == "small_city", col].dropna()

            strict = res[~res["city_class_imputed"]]
            metro_strict = strict.loc[strict["city_class"] == "metro", col].dropna()
            small_strict = strict.loc[strict["city_class"] == "small_city", col].dropna()

            rng = np.random.default_rng(42)
            m_samp = rng.choice(metro_area, size=min(5000, len(metro_area)), replace=False)
            s_samp = rng.choice(small_area, size=min(5000, len(small_area)), replace=False)
            sh_m = shapiro(m_samp)
            sh_s = shapiro(s_samp)

            u, p = mannwhitneyu(metro_area, small_area, alternative="less")
            if len(metro_strict) > 0 and len(small_strict) > 0:
                u_s, p_s = mannwhitneyu(metro_strict, small_strict, alternative="less")
            else:
                u_s, p_s = float("nan"), float("nan")

            plot_df = res.dropna(subset=[col]).sample(
                min(50000, len(res)), random_state=0
            )

            return {
                "n_metro": int(len(metro_area)),
                "n_small": int(len(small_area)),
                "mean_metro": float(metro_area.mean()),
                "mean_small": float(small_area.mean()),
                "shapiro_metro_W": float(sh_m.statistic),
                "shapiro_metro_p": float(sh_m.pvalue),
                "shapiro_small_W": float(sh_s.statistic),
                "shapiro_small_p": float(sh_s.pvalue),
                "U": float(u),
                "p": float(p),
                "strict_n_metro": int(len(metro_strict)),
                "strict_n_small": int(len(small_strict)),
                "strict_U": None if np.isnan(u_s) else float(u_s),
                "strict_p": None if np.isnan(p_s) else float(p_s),
                "plot_df": plot_df[[col, "city_class"]],
            }

        r1 = _run_h1(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Metro mean (m²)", f"{r1['mean_metro']:.1f}", f"n={r1['n_metro']:,}")
        c2.metric("Small mean (m²)", f"{r1['mean_small']:.1f}", f"n={r1['n_small']:,}")
        c3.metric("Mann-Whitney p", f"{r1['p']:.2e}")

        st.caption(
            f"Shapiro-Wilk — metro: W={r1['shapiro_metro_W']:.4f}, "
            f"p={r1['shapiro_metro_p']:.2e}  |  "
            f"small_city: W={r1['shapiro_small_W']:.4f}, "
            f"p={r1['shapiro_small_p']:.2e}"
        )

        col_left, col_right = st.columns(2)
        with col_left:
            fig = px.box(r1["plot_df"], x="city_class", y="building_size",
                         points=False,
                         labels={"city_class": "", "building_size": "building_size (m²)"},
                         title="Distribution of building_size")
            fig.update_layout(height=380, margin=dict(t=50, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            means = (r1["plot_df"].groupby("city_class", observed=True)["building_size"]
                     .mean().reset_index())
            fig = px.bar(means, x="city_class", y="building_size",
                         text="building_size",
                         labels={"city_class": "", "building_size": "mean building_size"},
                         title="Mean building_size per class")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(height=380, margin=dict(t=50, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Robustness check — strict (excluding imputed cities)**")
        if r1["strict_p"] is not None:
            st.write(
                f"metro: n={r1['strict_n_metro']:,}  |  "
                f"small_city: n={r1['strict_n_small']:,}  |  "
                f"p={r1['strict_p']:.2e}"
            )
        else:
            st.write("Strict groups empty — skipped.")

        if r1["p"] < 0.05 and r1["mean_metro"] < r1["mean_small"]:
            st.success(
                "Hypothesis **SUPPORTED**: metro houses are significantly smaller "
                f"(Mann-Whitney p = {r1['p']:.2e})."
            )
        else:
            st.info(f"Hypothesis **NOT supported** (Mann-Whitney p = {r1['p']:.2e}).")

# ==================================================================
# TAB 1 — H2: Old vs new home size
# ==================================================================
with tabs[1]:
    st.markdown("""
    **فرضیه:** «قدیما خونه‌ها دلبازتر بود!» — آیا میانگین مساحت خانه‌های قدیمی‌ساخت
    (قبل از ۱۳۹۶) بیشتر از خانه‌های جدیدساخت است؟
    """)

    homes = df[df['is_residential_unit'] & ~df['is_wanted_ad']].copy()
    homes = homes.dropna(subset=['construction_year', 'building_size'])
    homes['age_group'] = np.where(homes['construction_year'] < 1396, 'old', 'new')

    def compare_groups(old, new):
        t, p_t = stats.ttest_ind(old, new, equal_var=False, alternative='greater')
        u, p_u = stats.mannwhitneyu(old, new, alternative='greater')
        diff = old.mean() - new.mean()
        se = np.sqrt(old.var() / len(old) + new.var() / len(new))
        return pd.Series({'n_old': len(old), 'n_new': len(new),
                          'mean_old': old.mean(), 'mean_new': new.mean(),
                          'diff': diff, 'ci_low': diff - 1.96 * se,
                          'ci_high': diff + 1.96 * se,
                          'p_ttest': p_t,
                          'prob_old_larger': u / (len(old) * len(new)),
                          'p_mannwhitney': p_u})

    old = homes['age_group'] == 'old'
    result = compare_groups(homes.loc[old, 'building_size'],
                            homes.loc[~old, 'building_size']).round(2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Old mean (m²)", f"{homes.loc[old, 'building_size'].mean():.1f}",
              f"n={result['n_old']:,.0f}")
    c2.metric("New mean (m²)", f"{homes.loc[~old, 'building_size'].mean():.1f}",
              f"n={result['n_new']:,.0f}")
    c3.metric("Δ (old − new)", f"{result['diff']:.1f} m²")
    c4.metric("p (t-test, one-sided)", f"{result['p_ttest']:.2f}")

    st.info("p ≈ 1 means the data does **not** support the claim — new homes are in fact "
            "about 21 m² larger on average.")

    # Smooth KDE curves (matches the original notebook's gaussian_kde plot)
    x_grid = np.linspace(0, 400, 400)
    fig = go.Figure()
    for age, label, color, fill in [
        ('old', 'old', '#ff7f0e', 'rgba(255,127,14,0.35)'),
        ('new', 'new', '#1f77b4', 'rgba(31,119,180,0.35)'),
    ]:
        sizes = homes.loc[homes['age_group'] == age, 'building_size']
        sizes = sizes[(sizes > 0) & (sizes <= 400)].dropna()
        if len(sizes) > 1:
            density = gaussian_kde(sizes, bw_method=0.2)(x_grid)
            fig.add_trace(go.Scatter(
                x=x_grid, y=density, mode='lines',
                fill='tozeroy', name=label,
                line=dict(color=color, width=2),
                fillcolor=fill,
            ))
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=20, l=10, r=10),
        xaxis_title='Building size (m²)',
        yaxis_title='Density',
        title='Building size of old (before 1396) and new homes (≤ 400 m²)',
    )
    st.plotly_chart(fig, use_container_width=True)

    homes['kind'] = np.where(homes['cat3_slug'].str.startswith('apartment'),
                             'apartment', 'house_villa')
    by_year_all = (homes.groupby('construction_year')['building_size']
                   .mean().reset_index())
    by_year_kind = (homes.groupby(['construction_year', 'kind'])['building_size']
                    .mean().reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=by_year_all['construction_year'], y=by_year_all['building_size'],
        mode='lines+markers', name='All homes',
        line=dict(color='#000000', shape='spline'),
        line_shape='spline',
    ))
    for kind, color in [('apartment', '#4c72b0'),
                        ('house_villa', '#55a868')]:
        sub = by_year_kind[by_year_kind['kind'] == kind]
        fig.add_trace(go.Scatter(
            x=sub['construction_year'], y=sub['building_size'],
            mode='lines+markers', name=kind,
            line=dict(color=color, shape='spline'),
            line_shape='spline',
        ))
    fig.add_vline(x=1395.5, line_dash="dash", line_color="#d62728",
                  annotation_text="Cutoff (1396)")
    fig.update_layout(
        height=380,
        margin=dict(t=30, b=20, l=10, r=10),
        xaxis_title="Construction year",
        yaxis_title="Mean building size (m²)",
        yaxis_range=[0, 205],
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# TAB 2 — H3: Business deed effect on sale price
# ==================================================================
with tabs[2]:
    st.markdown("""
    **فرضیه:** آیا داشتن سند تجاری (`has_business_deed`) بر میانگین قیمت فروش ملک
    تجاری ارتباط معناداری دارد؟
    """)

    d = df[df['has_business_deed'].notna()].copy()
    with_deed = pd.to_numeric(d['has_business_deed'], errors='coerce')
    d = d.assign(_deed=with_deed)
    g1 = d.loc[d['_deed'] == 1, 'price_value'].dropna()
    g0 = d.loc[d['_deed'] == 0, 'price_value'].dropna()

    c1, c2 = st.columns(2)
    c1.metric("Mean with deed (Toman)", f"{g1.mean():,.0f}", f"n={len(g1):,}")
    c2.metric("Mean without deed (Toman)", f"{g0.mean():,.0f}", f"n={len(g0):,}")

    s1 = g1.sample(n=min(5000, len(g1)), random_state=44)
    s0 = g0.sample(n=min(5000, len(g0)), random_state=44)
    sw1 = shapiro(s1)
    sw0 = shapiro(s0)
    lev = levene(s1, s0)
    u, p = mannwhitneyu(s1, s0, alternative='two-sided')

    st.write(f"- Shapiro (with deed):    W = {sw1.statistic:.4f}, p = {sw1.pvalue:.2e}")
    st.write(f"- Shapiro (without deed): W = {sw0.statistic:.4f}, p = {sw0.pvalue:.2e}")
    # st.write(f"- Levene (equal variance): p = {lev.pvalue:.2e}")
    st.write(f"- Mann-Whitney: U = {u:,.0f}, **p = {p:.2e}**")

    if p < 0.05:
        st.success(
            "p < α (0.05): H₀ is rejected. Having a business deed **is** "
            "significantly associated with sale price."
        )
    else:
        st.info("p ≥ α: no significant association.")

# ==================================================================
# TAB 3 — H4: Luxury vs non-luxury amenities
# ==================================================================
with tabs[3]:
    st.markdown("""
    **فرضیه:** آیا وجود امکانات **لاکچری** (استخر، باربیکیو، سونا، جکوزی) و
    امکانات **غیرلاکچری** (پارکینگ، آسانسور، بالکن) میانه‌ی قیمت ملک‌های
    مسکونی را به‌طور معنادار تغییر می‌دهد؟
    """)

    @st.cache_data(show_spinner="Running H4 (Mann-Whitney on price_value) …")
    def _run_h4(df):
        p_col = "price_value"
        if p_col not in df.columns:
            return {"error": f"missing column: {p_col}"}

        data = df.copy()
        if "is_residential_unit" in data.columns:
            data = data[data["is_residential_unit"] == True]

        data = data[data[p_col].notna() & (data[p_col] > 0)].copy()
        if len(data) == 0:
            return {"error": "empty after price filter"}

        q_low, q_high = data[p_col].quantile(0.05), data[p_col].quantile(0.95)
        data = data[(data[p_col] >= q_low) & (data[p_col] <= q_high)].copy()

        lux_cols = ["has_pool_combined", "has_barbecue_combined",
                    "has_sauna_combined", "has_jacuzzi_combined"]
        non_lux_cols = ["has_parking", "has_elevator", "has_balcony_combined"]

        for c in lux_cols + non_lux_cols:
            if c in data.columns:
                data[c] = data[c].fillna(False).astype(bool)

        lux = [c for c in lux_cols if c in data.columns]
        non_lux = [c for c in non_lux_cols if c in data.columns]

        if not lux:
            return {"error": "no luxury columns found"}
        if not non_lux:
            return {"error": "no non-luxury columns found"}

        data["has_luxury"] = data[lux].any(axis=1)
        data["has_non_luxury"] = data[non_lux].any(axis=1)

        def _test(col):
            a = data.loc[data[col] == True, p_col]
            b = data.loc[data[col] == False, p_col]
            if len(a) == 0 or len(b) == 0:
                return None
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            med_a, med_b = float(a.median()), float(b.median())
            diff = ((med_a - med_b) / med_b) * 100 if med_b != 0 else float("nan")
            return {
                "n_a": int(len(a)), "n_b": int(len(b)),
                "med_a": med_a, "med_b": med_b,
                "diff_pct": float(diff),
                "U": float(u), "p": float(p),
            }

        return {"luxury": _test("has_luxury"),
                "non_luxury": _test("has_non_luxury")}

    r4 = _run_h4(df)

    if r4 is None or "error" in r4:
        st.warning("H4 skipped: " + (r4["error"] if r4 else "no result"))
    else:
        st.markdown("### امکانات لاکچری")
        r_lux = r4["luxury"]
        if r_lux is None:
            st.info("No data for the luxury group.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("با امکانات (n)", f"{r_lux['n_a']:,}")
            c2.metric("بدون امکانات (n)", f"{r_lux['n_b']:,}")
            c3.metric("میانه با امکانات", f"{r_lux['med_a']:,.0f}")
            c4.metric("میانه بدون امکانات", f"{r_lux['med_b']:,.0f}")
            st.write(f"درصد تغییر میانه: **{r_lux['diff_pct']:+.2f}%**  |  "
                     f"U = {r_lux['U']:,.0f}  |  **p = {r_lux['p']:.2e}**")
            if r_lux["p"] < 0.05:
                st.success("Reject H₀ — تفاوت معنادار است (p < 0.05).")
            else:
                st.info("Fail to reject H₀ — تفاوت معنادار نیست (p ≥ 0.05).")

        st.markdown("### امکانات غیرلاکچری")
        r_nlx = r4["non_luxury"]
        if r_nlx is None:
            st.info("No data for the non-luxury group.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("با امکانات (n)", f"{r_nlx['n_a']:,}")
            c2.metric("بدون امکانات (n)", f"{r_nlx['n_b']:,}")
            c3.metric("میانه با امکانات", f"{r_nlx['med_a']:,.0f}")
            c4.metric("میانه بدون امکانات", f"{r_nlx['med_b']:,.0f}")
            st.write(f"درصد تغییر میانه: **{r_nlx['diff_pct']:+.2f}%**  |  "
                     f"U = {r_nlx['U']:,.0f}  |  **p = {r_nlx['p']:.2e}**")
            if r_nlx["p"] < 0.05:
                st.success("Reject H₀ — تفاوت معنادار است (p < 0.05).")
            else:
                st.info("Fail to reject H₀ — تفاوت معنادار نیست (p ≥ 0.05).")

# ==================================================================
# TAB 4 — Binning (Apartment & Shop Size)
# ==================================================================
with tabs[4]:

    st.markdown("### ۱. دسته‌بندی متراژ آپارتمان")
    st.markdown("""
    متراژ آپارتمان یک عدد پیوسته است، ولی خریدار و مستأجر معمولاً با دسته‌هایی
    مثل «کوچک‌متراژ» یا «بزرگ» دنبال خانه می‌گردند. متراژ را به چهار دسته تبدیل
    می‌کنیم و بررسی می‌کنیم که قیمت هر متر در این دسته‌ها چه تغییری می‌کند.
    """)

    apt_req = ["cat3_slug", "city_slug", "building_size",
               "rooms_count", "price_per_m2"]
    apt_missing = [c for c in apt_req if c not in df.columns]

    if apt_missing:
        st.warning(f"ستون‌های مورد نیاز در parquet وجود ندارند: {apt_missing}")
    else:
        @st.cache_data(show_spinner="Binning apartment sizes …")
        def _run_apt_bin(df):
            d = df[apt_req].copy()
            d = d[d["cat3_slug"].isin(["apartment-sell", "apartment-rent"])].copy()
            d["relative_price"] = d["price_per_m2"] / d.groupby(
                ["city_slug", "cat3_slug"])["price_per_m2"].transform("median")

            labels = ["small (<60)", "medium (60-100)",
                      "large (100-150)", "very large (150+)"]
            d["size_class"] = pd.cut(d["building_size"],
                                     [0, 60, 100, 150, float("inf")],
                                     right=False, labels=labels)

            sale = d[d["cat3_slug"] == "apartment-sell"]
            tbl = sale.groupby("size_class", observed=True).agg(
                ads=("size_class", "size"),
                median_rooms=("rooms_count", "median"),
                median_relative_price=("relative_price", "median")).round(2)

            rows = []
            for cat in ["apartment-sell", "apartment-rent"]:
                part = d[d["cat3_slug"] == cat]
                groups = [g.dropna() for _, g in
                          part.groupby("size_class", observed=True)["relative_price"]]
                if len(groups) >= 2 and sum(len(g) for g in groups) > 0:
                    h, p = kruskal(*groups)
                    n = sum(len(g) for g in groups)
                    eps2 = (h - len(groups) + 1) / (n - len(groups))
                    medians = part.groupby("size_class", observed=True)[
                        "relative_price"].median().round(2).tolist()
                else:
                    p = eps2 = float("nan")
                    medians = []
                rows.append({"cat": cat, "p_value": float(p),
                             "epsilon_squared": float(eps2),
                             "medians": medians})

            plot_df = sale.dropna(subset=["relative_price", "size_class"]).sample(
                min(50000, len(sale)), random_state=0)[["size_class", "relative_price"]]

            return {"tbl": tbl, "rows": rows, "plot_df": plot_df,
                    "labels": labels}

        r_ab = _run_apt_bin(df)

        st.markdown("**آپارتمان‌های فروش — خلاصه به تفکیک دسته‌ی متراژ**")
        st.dataframe(r_ab["tbl"], use_container_width=True)

        fig = px.box(r_ab["plot_df"], x="size_class", y="relative_price",
                     category_orders={"size_class": r_ab["labels"]},
                     points=False,
                     labels={"size_class": "",
                             "relative_price": "Price per m2 / city median"},
                     title="Apartments for sale")
        fig.add_hline(y=1, line_dash="dash", line_color="#7f7f7f")
        fig.update_layout(height=400, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**آزمون Kruskal-Wallis**")
        for row in r_ab["rows"]:
            st.write(f"- `{row['cat']}`: p-value = {row['p_value']:.2e}, "
                     f"ε² = {row['epsilon_squared']:.2f}, "
                     f"medians = {row['medians']}")

    st.divider()

    st.markdown("### ۲. دسته‌بندی متراژ مغازه")
    st.markdown("""
    در مغازه‌ها چیزی که اهمیت دارد بیشتر دسترسی و ویترین است تا خود متراژ.
    اگر این حرف درست باشد، مغازه‌های کوچک‌تر باید قیمت هر متر بیشتری داشته باشند.
    """)

    shop_req = ["cat3_slug", "city_slug", "building_size", "price_per_m2"]
    shop_missing = [c for c in shop_req if c not in df.columns]

    if shop_missing:
        st.warning(f"ستون‌های مورد نیاز در parquet وجود ندارند: {shop_missing}")
    else:
        @st.cache_data(show_spinner="Binning shop sizes …")
        def _run_shop_bin(df):
            d = df[shop_req].copy()
            d = d[d["cat3_slug"].isin(["shop-sell", "shop-rent"])].copy()
            d["relative_price"] = d["price_per_m2"] / d.groupby(
                ["city_slug", "cat3_slug"])["price_per_m2"].transform("median")
            d["size_class"] = pd.cut(d["building_size"],
                                     [0, 20, 50, 100, float("inf")],
                                     right=False,
                                     labels=["<20", "20-50", "50-100", "100+"])

            tbl = d.pivot_table(index="size_class", columns="cat3_slug",
                                values="relative_price",
                                aggfunc=["count", "median"],
                                observed=True).round(2)

            def kruskal_eps(part):
                groups = [g.dropna() for _, g in
                          part.groupby("size_class", observed=True)["relative_price"]]
                if len(groups) < 2 or sum(len(g) for g in groups) == 0:
                    return float("nan"), float("nan")
                h, p = kruskal(*groups)
                n = sum(len(g) for g in groups)
                return float(p), float((h - len(groups) + 1) / (n - len(groups)))

            tests = {}
            for cat in ["shop-sell", "shop-rent"]:
                tests[cat] = kruskal_eps(d[d["cat3_slug"] == cat])

            return {"tbl": tbl, "tests": tests}

        r_sb = _run_shop_bin(df)

        st.markdown("**خلاصه: تعداد و میانه قیمت نسبی هر متر**")
        st.dataframe(r_sb["tbl"], use_container_width=True)

        med = r_sb["tbl"]["median"] if "median" in r_sb["tbl"].columns else None
        if med is not None and not med.empty:
            plot_df = med.reset_index().melt(id_vars="size_class",
                                             var_name="deal", value_name="median")
            plot_df["deal"] = plot_df["deal"].map(
                {"shop-rent": "Rent", "shop-sell": "Sale"})
            fig = px.bar(plot_df, x="size_class", y="median", color="deal",
                         barmode="group",
                         labels={"size_class": "Shop size (m²)",
                                 "median": "Median price per m2 / city median",
                                 "deal": "Deal"},
                         title="Median relative price by shop size")
            fig.add_hline(y=1, line_dash="dash", line_color="#7f7f7f")
            fig.update_layout(height=350, margin=dict(t=50, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**آزمون Kruskal-Wallis**")
        for cat, (p, eps2) in r_sb["tests"].items():
            st.write(f"- `{cat}`: p-value = {p:.2e}, ε² = {eps2:.3f}")

# ==================================================================
# TAB 5 — Reclassification (Climate, Moving Season, Tourism)
# ==================================================================
with tabs[5]:

    st.markdown("### ۱. اقلیم از روی سیستم سرمایشی")
    st.markdown("""
    اطلاعات اقلیم در دیتاست نیست، ولی نوع سیستم سرمایشی خانه‌ها نشانه‌ی خوبی از آن است:
    در هوای شرجی کولر آبی کار نمی‌کند و مردم اسپلیت می‌گذارند. شهرها را با همین نشانه
    به سه گروه اقلیمی تقسیم می‌کنیم و بعد بررسی می‌کنیم که آیا ارزش سرمایش مرکزی در
    همه‌ی اقلیم‌ها یکسان است یا نه.
    """)

    climate_req = ["cat3_slug", "city_slug", "has_cooling_system", "price_per_m2"]
    climate_missing = [c for c in climate_req if c not in df.columns]

    if climate_missing:
        st.warning(f"ستون‌های مورد نیاز در parquet وجود ندارند: {climate_missing}")
    else:
        @st.cache_data(show_spinner="Climate reclassification …")
        def _run_climate(df):
            d = df[climate_req].copy()
            homes = d[d["cat3_slug"].astype("string").str.startswith(
                ("apartment", "house"), na=False)].copy()
            homes = homes[homes["has_cooling_system"].isin(
                ["split", "air_conditioner", "water_cooler"])]

            city = homes.groupby("city_slug")["has_cooling_system"].agg(
                ads="size",
                split_share=lambda s: (s != "water_cooler").mean())
            city = city[city["ads"] >= 150].copy()
            city["climate"] = pd.cut(city["split_share"], [0, 0.3, 0.7, 1],
                                     labels=["dry", "mixed", "humid"],
                                     include_lowest=True)

            d["climate"] = d["city_slug"].map(city["climate"])

            apt = d[d["cat3_slug"] == "apartment-sell"].copy()
            apt["relative_price"] = apt["price_per_m2"] / apt.groupby(
                "city_slug")["price_per_m2"].transform("median")
            apt["cooling"] = apt["has_cooling_system"].map({
                "water_cooler": "evaporative",
                "split": "split",
                "air_conditioner": "split",
                "duct_split": "central",
                "fan_coil": "central",
            })

            pivot = apt.pivot_table(index="climate", columns="cooling",
                                    values="relative_price",
                                    aggfunc="median", observed=True)

            rows = []
            for cl in ["dry", "mixed", "humid"]:
                part = apt[apt["climate"] == cl]
                central = part.loc[part["cooling"] == "central",
                                   "relative_price"].dropna()
                split = part.loc[part["cooling"] == "split",
                                 "relative_price"].dropna()
                if len(central) > 0 and len(split) > 0:
                    u, p = mannwhitneyu(central, split)
                    r = 2 * u / (len(central) * len(split)) - 1
                else:
                    p, r = float("nan"), float("nan")
                rows.append({"climate": cl,
                             "n_central": int(len(central)),
                             "n_split": int(len(split)),
                             "p_value": float(p),
                             "r": float(r)})

            return {"city": city.reset_index(),
                    "pivot": pivot,
                    "tests": pd.DataFrame(rows)}

        r_clim = _run_climate(df)
        city_df = r_clim["city"]

        fig = px.histogram(city_df, x="split_share", nbins=20,
                           labels={"split_share": "Share of split / AC",
                                   "count": "Number of cities"},
                           title="Split/AC share across cities")
        fig.add_vline(x=0.3, line_dash="dash", line_color="#d62728")
        fig.add_vline(x=0.7, line_dash="dash", line_color="#d62728")
        fig.update_layout(height=350, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        بیشتر شهرها یا تقریباً همه کولر آبی دارند یا تقریباً همه اسپلیت، و تعداد کمی
        بین این دو هستند. برای همین مرزها را ۰.۳ و ۰.۷ گذاشتیم:

        - `dry`: سهم اسپلیت تا ۰.۳
        - `mixed`: بین ۰.۳ و ۰.۷
        - `humid`: بیشتر از ۰.۷
        """)

        st.markdown("**تعداد شهرها در هر اقلیم**")
        st.dataframe(city_df["climate"].value_counts()
                     .rename_axis("climate").to_frame("count"))

        st.markdown("**میانه قیمت نسبی هر متر (آپارتمان فروش)**")
        st.dataframe(r_clim["pivot"].round(2), use_container_width=True)

        st.markdown("**آزمون Mann-Whitney: سرمایش مرکزی در مقابل اسپلیت**")
        st.dataframe(r_clim["tests"].round(4), use_container_width=True)

    st.divider()

    st.markdown("### ۲. فصل جابه‌جایی")
    st.markdown("""
    معمولاً گفته می‌شود که بازار اجاره در تابستان داغ می‌شود، چون خانواده‌ها
    می‌خواهند قبل از شروع مدرسه‌ها جابه‌جا شوند. ماه ثبت آگهی را به سه دوره تبدیل
    می‌کنیم و بررسی می‌کنیم که این دوره‌ها روی عرضه‌ی اجاره و قیمت آن اثر دارند یا نه.
    """)

    season_req = ["cat3_slug", "city_slug", "jalali_month",
                  "in_main_window", "price_per_m2"]
    season_missing = [c for c in season_req if c not in df.columns]

    if season_missing:
        st.warning(f"ستون‌های مورد نیاز در parquet وجود ندارند: {season_missing}")
    else:
        @st.cache_data(show_spinner="Moving-season reclassification …")
        def _run_moving_season(df):
            d = df[season_req].copy()
            apt = d[d["in_main_window"] &
                    d["cat3_slug"].isin(["apartment-sell", "apartment-rent"])].copy()

            season = {
                "1403-02 Ordibehesht": "pre_season",
                "1403-03 Khordad": "moving_season",
                "1403-04 Tir": "moving_season",
                "1403-05 Mordad": "moving_season",
                "1403-06 Shahrivar": "moving_season",
                "1403-07 Mehr": "post_season",
                "1403-08 Aban": "post_season",
                "1403-09 Azar": "post_season",
            }
            apt["season"] = apt["jalali_month"].map(season)

            monthly = pd.crosstab(apt["jalali_month"], apt["cat3_slug"])
            if "apartment-rent" in monthly.columns:
                monthly["rent_share_%"] = (
                    monthly["apartment-rent"] / monthly.sum(axis=1) * 100
                ).round(1)

            table = pd.crosstab(apt["season"], apt["cat3_slug"])
            if table.size > 0:
                chi2, p_supply, dof, _ = chi2_contingency(table)
                v_supply = np.sqrt(chi2 / (table.values.sum() *
                                           (min(table.shape) - 1)))
                supply_share = (pd.crosstab(apt["season"], apt["cat3_slug"],
                                            normalize="index") * 100).round(1)
            else:
                p_supply = v_supply = float("nan")
                supply_share = pd.DataFrame()

            rent = apt[apt["cat3_slug"] == "apartment-rent"].copy()
            rent["relative_price"] = rent["price_per_m2"] / rent.groupby(
                "city_slug")["price_per_m2"].transform("median")
            summary = rent.groupby("season").agg(
                raw_median_million=("price_per_m2", lambda s: s.median() / 1e6),
                relative_median=("relative_price", "median"))

            groups = [g.dropna() for _, g in
                      rent.groupby("season")["relative_price"]]
            if len(groups) >= 2 and sum(len(g) for g in groups) > 0:
                h, p_price = kruskal(*groups)
                n = sum(len(g) for g in groups)
                eps2 = (h - len(groups) + 1) / (n - len(groups))
            else:
                p_price = eps2 = float("nan")

            return {"monthly": monthly,
                    "supply_share": supply_share,
                    "p_supply": float(p_supply),
                    "v_supply": float(v_supply),
                    "summary": summary,
                    "p_price": float(p_price),
                    "eps2": float(eps2)}

        r_ms = _run_moving_season(df)

        st.markdown("**آگهی‌های آپارتمان به تفکیک ماه**")
        st.dataframe(r_ms["monthly"], use_container_width=True)

        if not r_ms["monthly"].empty:
            plot_df = r_ms["monthly"].reset_index()
            plot_df["month_label"] = plot_df["jalali_month"].str.split().str[1]

            fig = go.Figure()
            if "apartment-rent" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df["month_label"],
                                         y=plot_df["apartment-rent"],
                                         mode="lines+markers", name="Rent",
                                         line=dict(color="#1f77b4",
                                                   shape="spline"),
                                         line_shape="spline"))
            if "apartment-sell" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df["month_label"],
                                         y=plot_df["apartment-sell"],
                                         mode="lines+markers", name="Sale",
                                         line=dict(color="#ff7f0e",
                                                   shape="spline"),
                                         line_shape="spline"))
            fig.add_vrect(x0=0.5, x1=4.5, fillcolor="#ffa500", opacity=0.15,
                          line_width=0)
            fig.update_layout(height=350,
                              margin=dict(t=30, b=20, l=10, r=10),
                              yaxis_title="Apartment ads")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**عرضه — سهم اجاره از کل آپارتمان‌ها به تفکیک دوره**")
        if not r_ms["supply_share"].empty:
            st.dataframe(r_ms["supply_share"], use_container_width=True)
        st.write(f"Chi² p-value = {r_ms['p_supply']:.2e}  |  "
                 f"Cramér's V = {r_ms['v_supply']:.2f}")

        st.markdown("**قیمت — میانه قیمت هر متر اجاره**")
        st.dataframe(r_ms["summary"].round(3), use_container_width=True)
        st.write(f"Kruskal-Wallis p-value = {r_ms['p_price']:.2e}  |  "
                 f"ε² = {r_ms['eps2']:.4f}")

    st.divider()

    st.markdown("### ۳. نوع گردشگری شهرها")
    st.markdown("""
    تعداد شهرها در `city_slug` زیاد است و بررسی تک‌تک آن‌ها برای بازار اجاره‌ی
    کوتاه‌مدت کمکی نمی‌کند. شهرها را بر اساس هدف اصلی سفر به ۴ گروه تقسیم
    می‌کنیم و بررسی می‌کنیم که آیا نوع گردشگری روی نوع اقامتگاه و قیمت آن اثر
    دارد یا نه.
    """)

    tourism_req = ["cat2_slug", "cat3_slug", "city_slug",
                   "location_latitude", "location_longitude",
                   "regular_person_capacity", "rent_price_on_regular_days"]
    tourism_missing = [c for c in tourism_req if c not in df.columns]

    if tourism_missing:
        st.warning(f"ستون‌های مورد نیاز در parquet وجود ندارند: {tourism_missing}")
    else:
        @st.cache_data(show_spinner="Tourism reclassification …")
        def _run_tourism(df):
            d = df[tourism_req].copy()

            pilgrimage = ["mashhad", "qom"]
            cultural = ["isfahan", "shiraz", "yazd", "kashan", "tabriz"]
            south_islands = ["kish", "qeshm", "chabahar", "hormuz"]

            centers = d.groupby("city_slug")[
                ["location_latitude", "location_longitude"]].median()
            north = centers[(centers["location_latitude"] > 36.3) &
                            (centers["location_longitude"].between(48.7, 54.7))].index

            def tourism_type(city):
                if city in pilgrimage:
                    return "pilgrimage"
                if city in cultural:
                    return "cultural"
                if city in north or city in south_islands:
                    return "leisure"
                return "non_tourist"

            d["tourism_type"] = d["city_slug"].map(tourism_type,
                                                   na_action="ignore")

            daily = d[d["cat2_slug"] == "temporary-rent"].copy()
            daily["price_per_person"] = (daily["rent_price_on_regular_days"] /
                                         daily["regular_person_capacity"])
            suites = daily[daily["cat3_slug"] == "suite-apartment"].dropna(
                subset=["price_per_person"])

            summary = pd.crosstab(daily["tourism_type"], daily["cat3_slug"],
                                  normalize="index") * 100
            summary["ads"] = daily["tourism_type"].value_counts()
            summary["suite price per person (k toman)"] = (
                suites.groupby("tourism_type")["price_per_person"].median() / 1000)

            table = pd.crosstab(daily["tourism_type"], daily["cat3_slug"])
            if table.size > 0:
                chi2, p_h1, dof, _ = chi2_contingency(table)
                v_h1 = np.sqrt(chi2 / (table.values.sum() *
                                       (min(table.shape) - 1)))
            else:
                p_h1 = v_h1 = float("nan")

            groups = [g["price_per_person"] for _, g in
                      suites.groupby("tourism_type")]
            if len(groups) >= 2 and len(suites) > 0:
                h, p_h2 = kruskal(*groups)
                eps2_h2 = (h - len(groups) + 1) / (len(suites) - len(groups))
            else:
                p_h2 = eps2_h2 = float("nan")

            is_pil = suites["tourism_type"] == "pilgrimage"
            if is_pil.sum() > 0 and (~is_pil).sum() > 0:
                u, p_pil = mannwhitneyu(
                    suites.loc[is_pil, "price_per_person"],
                    suites.loc[~is_pil, "price_per_person"])
                r_pil = 1 - 2 * u / (is_pil.sum() * (~is_pil).sum())
            else:
                p_pil = r_pil = float("nan")

            return {"summary": summary,
                    "p_h1": float(p_h1), "v_h1": float(v_h1),
                    "p_h2": float(p_h2), "eps2_h2": float(eps2_h2),
                    "p_pil": float(p_pil), "r_pil": float(r_pil)}

        r_t = _run_tourism(df)

        st.markdown("**خلاصه: نوع اقامتگاه و قیمت به تفکیک نوع گردشگری**")
        st.dataframe(r_t["summary"].round(1), use_container_width=True)

        suite_villa = [c for c in ["suite-apartment", "villa"]
                       if c in r_t["summary"].columns]
        if suite_villa:
            plot_df = (r_t["summary"][suite_villa].reset_index()
                       .melt(id_vars="tourism_type", var_name="kind",
                             value_name="pct"))
            fig = px.bar(plot_df, x="tourism_type", y="pct", color="kind",
                         barmode="group",
                         labels={"pct": "% of short-stay ads",
                                 "tourism_type": "",
                                 "kind": "Accommodation"},
                         title="Suites vs villas by tourism type")
            fig.update_layout(height=380,
                              margin=dict(t=50, b=20, l=10, r=10),
                              yaxis_range=[0, 90])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**H1: نوع گردشگری و نوع اقامتگاه (Chi²)**")
        st.write(f"p-value = {r_t['p_h1']:.2e}  |  "
                 f"Cramér's V = {r_t['v_h1']:.2f}")
        if r_t["p_h1"] < 0.05:
            st.success("Reject H₀ — نوع اقامتگاه به نوع گردشگری وابسته است.")
        else:
            st.info("Fail to reject H₀ — استقلال برقرار است.")

        st.markdown("**H2: قیمت هر نفر در سوئیت‌ها (Kruskal-Wallis)**")
        st.write(f"p-value = {r_t['p_h2']:.2e}  |  "
                 f"ε² = {r_t['eps2_h2']:.3f}")

        st.markdown("**مقایسه‌ی شهرهای زیارتی با بقیه (Mann-Whitney)**")
        st.write(f"p-value = {r_t['p_pil']:.2e}  |  "
                 f"rank-biserial r = {r_t['r_pil']:.2f}")

# ==================================================================
# TAB 6 — Bonus #1
# ==================================================================
with tabs[6]:
    st.markdown("""
    **آزمون امتیازی — نسبت دو نمونه‌ای:**
    آیا نسبت خانه‌های دارای **آسانسور** بین خانه‌های **بازسازی‌شده** و
    **بازسازی‌نشده** تفاوت معناداری دارد؟
    """)

    required = ["has_elevator", "is_rebuilt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in `divar_clean.parquet`: {missing}")
        st.stop()

    @st.cache_data(show_spinner="Running Bonus #1 …")
    def _run_bonus1(df):
        d = df[required].dropna()
        if len(d) == 0:
            return {"error": "empty after dropna"}

        d = d.assign(
            has_elevator=pd.to_numeric(d["has_elevator"], errors="coerce"),
            is_rebuilt=pd.to_numeric(d["is_rebuilt"], errors="coerce"),
        ).dropna()

        rebuilt = d[d["is_rebuilt"] == 1]
        not_rebuilt = d[d["is_rebuilt"] == 0]
        n_r, n_nr = int(len(rebuilt)), int(len(not_rebuilt))
        if n_r == 0 or n_nr == 0:
            return {"error": "one of the groups is empty"}

        e_r = int((rebuilt["has_elevator"] == 1).sum())
        e_nr = int((not_rebuilt["has_elevator"] == 1).sum())
        p_r = e_r / n_r
        p_nr = e_nr / n_nr

        z, p = proportions_ztest(count=[e_r, e_nr],
                                 nobs=[n_r, n_nr],
                                 alternative="two-sided")
        return {"n_rebuilt": n_r, "n_not_rebuilt": n_nr,
                "elev_rebuilt": e_r, "elev_not_rebuilt": e_nr,
                "p_rebuilt": float(p_r), "p_not_rebuilt": float(p_nr),
                "z": float(z), "p": float(p)}

    rb = _run_bonus1(df)

    if rb is None or "error" in rb:
        st.warning("Bonus #1 skipped: " + (rb["error"] if rb else "no result"))
    else:
        c1, c2 = st.columns(2)
        c1.metric("Rebuilt — share with elevator",
                  f"{rb['p_rebuilt']*100:.1f}%",
                  f"n={rb['n_rebuilt']:,}")
        c2.metric("Not Rebuilt — share with elevator",
                  f"{rb['p_not_rebuilt']*100:.1f}%",
                  f"n={rb['n_not_rebuilt']:,}")

        st.write(f"**Two-sample proportion Z-test:** z = {rb['z']:.3f}, "
                 f"p = {rb['p']:.3e}")
        if rb["p"] < 0.05:
            st.success("Reject H₀ — elevator proportion differs significantly.")
        else:
            st.info("Fail to reject H₀ — no significant difference.")

        plot_df = pd.DataFrame({
            "group": ["Rebuilt", "Not Rebuilt"],
            "pct": [rb["p_rebuilt"] * 100, rb["p_not_rebuilt"] * 100],
        })
        fig = px.bar(plot_df, x="group", y="pct", text="pct",
                     labels={"group": "", "pct": "Elevator proportion (%)"},
                     title="Elevator Proportion by Rebuilt Status")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=350,
                          margin=dict(t=50, b=20, l=10, r=10),
                          yaxis_range=[0, max(plot_df["pct"]) * 1.2])
        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# TAB 7 — Bonus #2
# ==================================================================
with tabs[7]:
    st.markdown("""
    **آزمون امتیازی — Kruskal-Wallis:**
    آیا میانه‌ی قیمت بین **جهت‌های مختلف ساختمان** تفاوت معناداری دارد؟
    """)

    required = ["price_value", "building_direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in `divar_clean.parquet`: {missing}")
        st.stop()

    @st.cache_data(show_spinner="Running Bonus #2 …")
    def _run_bonus2(df):
        d = df[required].copy()
        d = d.dropna(subset=["price_value", "building_direction"])
        d["price_value"] = pd.to_numeric(d["price_value"], errors="coerce")
        d = d.dropna(subset=["price_value"])

        q1 = d["price_value"].quantile(0.25)
        q3 = d["price_value"].quantile(0.75)
        iqr = q3 - q1
        d["price_value"] = d["price_value"].clip(lower=q1 - 1.5 * iqr,
                                                 upper=q3 + 1.5 * iqr)

        groups = [d.loc[d["building_direction"] == dr, "price_value"]
                  for dr in d["building_direction"].unique()]
        if len(groups) < 2:
            return {"error": "need at least two groups"}

        stat, p_value = kruskal(*groups)
        summary = (d.groupby("building_direction")["price_value"]
                     .agg(count="count", mean="mean", median="median",
                          min="min", max="max")
                     .sort_values("median", ascending=False))
        return {"stat": float(stat), "p": float(p_value),
                "summary": summary, "n_rows": int(len(d))}

    r2 = _run_bonus2(df)
    if "error" in r2:
        st.warning("Bonus #2 skipped: " + r2["error"])
    else:
        st.caption(f"Rows after cleaning: {r2['n_rows']:,}")
        st.write(f"**Kruskal-Wallis:** H = {r2['stat']:.2f}, p = {r2['p']:.3e}")
        if r2["p"] < 0.05:
            st.success("Reject H₀ — significant difference between the groups.")
        else:
            st.info("Fail to reject H₀ — no significant difference.")

        st.markdown("**Price summary by building direction**")
        st.dataframe(r2["summary"].round(0), use_container_width=True)

        fig = px.bar(r2["summary"].reset_index(),
                     x="building_direction", y="median", text="median",
                     title="Median Price by Building Direction",
                     labels={"median": "Median Price Value",
                             "building_direction": "Building Direction"})
        fig.update_traces(texttemplate="%{text:.3s}", textposition="outside")
        fig.update_layout(height=350, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# TAB 8 — Bonus #3
# ==================================================================
with tabs[8]:
    st.markdown("""
    **آزمون امتیازی — کای دو استقلال:**
    آیا **نوع ملک** (`property_type`) و **نوع سند** (`deed_type`) به هم وابسته‌اند؟
    """)

    required = ["property_type", "deed_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in `divar_clean.parquet`: {missing}")
        st.stop()

    @st.cache_data(show_spinner="Running Bonus #3 …")
    def _run_bonus3(df):
        d = df[required].copy().dropna(subset=required)
        table = pd.crosstab(d["property_type"], d["deed_type"])
        if table.size == 0:
            return {"error": "empty contingency table"}

        chi2, p, dof, expected = chi2_contingency(table)
        cells_lt5 = int((expected < 5).sum())
        total_cells = int(expected.size)
        pct_lt5 = cells_lt5 / total_cells * 100
        summary = (pd.crosstab(d["property_type"], d["deed_type"],
                               normalize="index") * 100)
        return {"table": table, "chi2": float(chi2), "p": float(p),
                "dof": int(dof), "cells_lt5": cells_lt5,
                "total_cells": total_cells, "pct_lt5": float(pct_lt5),
                "summary": summary}

    r3 = _run_bonus3(df)
    if "error" in r3:
        st.warning("Bonus #3 skipped: " + r3["error"])
    else:
        st.markdown("**Contingency table**")
        st.dataframe(r3["table"], use_container_width=True)

        st.write(f"- Total cells: {r3['total_cells']}")
        st.write(f"- Cells with expected count < 5: {r3['cells_lt5']} "
                 f"({r3['pct_lt5']:.1f}%)")

        if r3["pct_lt5"] <= 20:
            st.write(f"**Chi-Square:** χ² = {r3['chi2']:.2f}, "
                     f"dof = {r3['dof']}, p = {r3['p']:.3e}")
            if r3["p"] < 0.05:
                st.success("Reject H₀ — property_type and deed_type are **dependent**.")
            else:
                st.info("Fail to reject H₀ — independent.")
        else:
            st.warning("Chi-Square conditions not met.")

        st.markdown("**Deed type % distribution by property type**")
        st.dataframe(r3["summary"].round(2), use_container_width=True)

        plot_df = (r3["summary"].reset_index()
                   .melt(id_vars="property_type",
                         var_name="deed_type", value_name="pct"))
        fig = px.bar(plot_df, x="property_type", y="pct", color="deed_type",
                     title="Deed Type Proportion by Property Type",
                     labels={"pct": "Proportion (%)",
                             "property_type": "Property Type"})
        fig.update_layout(barmode="stack", yaxis_range=[0, 100],
                          height=400, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)