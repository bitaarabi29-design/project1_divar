import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, levene
from data_utils import load_data

st.set_page_config(page_title="Hypothesis Tests", layout="wide")
st.title("🧪 Hypothesis Tests")

df = load_data()
df['created_at_month'] = pd.to_datetime(df['created_at_month'], errors='coerce')

tabs = st.tabs([
    "H1 — Metro vs small-city size",
    "H2 — Old vs new home size",
    "H3 — Business deed & sale price",
    "H4 — Luxury vs non-luxury amenities",
    "Bonus #1 — Elevator & rebuild (Z-test)",
    "Bonus #2 — Building direction & price (Kruskal)",
    "Bonus #3 — Property & deed type (Chi²)",
])

# ==================================================================
# H1 — Metro houses smaller than small-city houses?
# ==================================================================
with tabs[0]:
    st.markdown("""
    **فرضیه:** با توجه به رشد مهاجرت به کلان‌شهرها و تراکم جمعیت، میانگین مساحت خانه‌های مسکونی
    در کلان‌شهرها نسبت به شهرهای کوچک کمتر است.
    """)

    def clean_building_size(d, col='building_size', min_limit=20, max_limit=2500):
        hard = d[col].notna() & ~d[col].between(min_limit, max_limit)
        d.loc[hard, col] = pd.NA
        positive = d[col].where(d[col] > 0)
        log_vals = np.log10(positive)
        q1, q3 = log_vals.quantile(0.25), log_vals.quantile(0.75)
        iqr = q3 - q1
        low, high = 10 ** (q1 - 2.0 * iqr), 10 ** (q3 + 2.0 * iqr)
        soft = d[col].notna() & ~d[col].between(low, high)
        d.loc[soft, col] = pd.NA
        return d

    residential_cats = ["apartment-sell", "apartment-rent",
                        "house-villa-sell", "house-villa-rent",
                        "villa", "suite-apartment"]
    res = df[df["cat3_slug"].astype("string").isin(residential_cats)].copy()
    res["building_size"] = pd.to_numeric(res["building_size"], errors="coerce")
    res = clean_building_size(res, col='building_size', min_limit=20, max_limit=2500)

    # is_metro already exists in the parquet (Step 11 of preprocessing)
    res = res.dropna(subset=['building_size', 'is_metro'])
    metro_area = res.loc[res["is_metro"] == True, "building_size"]
    small_area = res.loc[res["is_metro"] == False, "building_size"]

    t_stat, p_one_t = ttest_ind(metro_area, small_area, equal_var=False, alternative="less")
    u_stat, p_one_mw = mannwhitneyu(metro_area, small_area, alternative="less")

    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        return (x.mean() - y.mean()) / pooled

    d_eff = cohen_d(metro_area, small_area)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Metro mean (m²)", f"{metro_area.mean():.1f}", f"n={len(metro_area):,}")
    c2.metric("Small mean (m²)", f"{small_area.mean():.1f}", f"n={len(small_area):,}")
    c3.metric("Welch p", f"{p_one_t:.2e}")
    c4.metric("Cohen's d", f"{d_eff:.3f}")

    supported = (t_stat < 0) and (p_one_t < 0.05) and (p_one_mw < 0.05)
    st.success(f"Hypothesis **{'SUPPORTED' if supported else 'NOT supported'}** "
               f"(Mann-Whitney p = {p_one_mw:.2e})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    res.boxplot(column="building_size", by="is_metro", ax=axes[0], showfliers=False)
    axes[0].set_title("Distribution of building_size")
    axes[0].set_xlabel("is_metro")
    axes[0].set_ylabel("building_size (m²)")
    means = res.groupby("is_metro")["building_size"].mean()
    means.plot(kind="bar", ax=axes[1], color=["#4c72b0", "#dd8452"])
    axes[1].set_title("Mean building_size per group")
    axes[1].set_ylabel("mean building_size")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ==================================================================
# H2 — "Old houses were wider" (pre-1396)
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
                          'diff': diff, 'ci_low': diff - 1.96 * se, 'ci_high': diff + 1.96 * se,
                          'p_ttest': p_t, 'prob_old_larger': u / (len(old) * len(new)),
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

    bins = np.logspace(1, 3, 31)
    fig, ax = plt.subplots(figsize=(10, 4))
    for group, color in [('old', 'tab:orange'), ('new', 'tab:blue')]:
        sizes = homes.loc[homes['age_group'] == group, 'building_size']
        ax.hist(sizes, bins=bins, weights=np.ones(len(sizes)) / len(sizes),
                histtype='step', linewidth=1.5, color=color,
                label=f'{group} (mean {sizes.mean():.0f} m²)')
        ax.axvline(sizes.mean(), color=color, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Building size (m², log scale)')
    ax.set_ylabel('Share of ads in the group')
    ax.set_title('Building size of old (before 1396) and new homes')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Age curve
    homes['kind'] = np.where(homes['cat3_slug'].str.startswith('apartment'),
                             'apartment', 'house_villa')
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(homes.groupby('construction_year')['building_size'].mean(),
            marker='o', color='black', label='All homes')
    for kind, color in [('apartment', 'tab:blue'), ('house_villa', 'tab:green')]:
        by_year = homes[homes['kind'] == kind].groupby('construction_year')['building_size'].mean()
        ax.plot(by_year, marker='.', color=color, label=kind)
    ax.axvline(1395.5, color='red', linestyle='--', label='Cutoff (1396)')
    ax.set_xlabel('Construction year (1370 means 1370 or earlier)')
    ax.set_ylabel('Mean building size (m²)')
    ax.set_ylim(top=205)
    ax.set_title('Mean building size by construction year')
    ax.legend(loc='upper center', ncol=4)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ==================================================================
# H3 — Business deed effect on sale price
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
    st.write(f"- Levene (equal variance): p = {lev.pvalue:.2e}")
    st.write(f"- Mann-Whitney: U = {u:,.0f}, **p = {p:.2e}**")

    if p < 0.05:
        st.success(
            "p < α (0.05): H₀ is rejected. Having a business deed **is** "
            "significantly associated with sale price."
        )
    else:
        st.info("p ≥ α: no significant association.")

# ==================================================================
# H4 — Luxury vs non-luxury amenities on sale price (Tehran only)
# ==================================================================
with tabs[3]:
    st.markdown("""
    **فرضیه:** آیا وجود امکانات **لاکچری** (استخر) یا **غیرلاکچری** (پارکینگ)
    میانگین قیمت فروش را در تهران به‌طور معنادار تغییر می‌دهد؟
    """)

    from pathlib import Path

    csv_path = Path(__file__).parent.parent / "data" / "final_data_slim.csv"
    if not csv_path.exists():
        st.error(f"File not found: {csv_path}")
        st.stop()

    df4 = pd.read_csv(csv_path, low_memory=False)

    # فقط تهران
    df4 = df4[df4["city_slug"] == "tehran"].copy()

    # فقط فروش
    df4 = df4[df4["deal_family"] == "sell"].copy()

    # تمیز کردن قیمت
    price_col = "price_value"
    df4 = df4[df4[price_col].notna() & (df4[price_col] > 0)]
    low, high = df4[price_col].quantile([0.01, 0.99])
    df4 = df4[(df4[price_col] >= low) & (df4[price_col] <= high)]

    def mw_test(d, col, name):
        if col not in d.columns:
            st.info(f"Column `{col}` not found in data — skipping **{name}**.")
            return
        sub = d[d[col].notna()].copy()
        flag = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.assign(_f=flag)
        g1 = sub.loc[sub["_f"] == 1, price_col]
        g0 = sub.loc[sub["_f"] == 0, price_col]
        if len(g1) < 10 or len(g0) < 10:
            st.warning(f"{name}: not enough data")
            return

        u, p = mannwhitneyu(g1, g0, alternative="two-sided")
        med1, med0 = g1.median(), g0.median()
        diff = (med1 - med0) / med0 * 100

        with st.expander(f"**{name}**", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("median (has)", f"{med1:,.0f}", f"n={len(g1):,}")
            c2.metric("median (no)", f"{med0:,.0f}", f"n={len(g0):,}")
            c3.metric("diff %", f"{diff:+.1f}%")
            st.write(f"**p = {p:.3g}**")
            if p < 0.05:
                if abs(diff) >= 100:
                    st.write("→ تفاوت معنادار و **بزرگ**")
                elif abs(diff) >= 30:
                    st.write("→ تفاوت معنادار و **متوسط**")
                elif abs(diff) >= 10:
                    st.write("→ تفاوت معنادار ولی **کوچک**")
                else:
                    st.write("→ تفاوت معنادار ولی **ناچیز**")
            else:
                st.write("→ تفاوت **غیرمعنادار**")

    st.subheader("Luxury amenity")
    mw_test(df4, "is_pool_tehran", "استخر (لاکچری)")

    st.subheader("Non-luxury amenity")
    mw_test(df4, "is_parking_tehran", "پارکینگ (غیرلاکچری)")

# ==================================================================
# Bonus #1 — Two-Sample Proportion Z-Test: Elevator & Rebuilt
# ==================================================================
with tabs[4]:
    st.markdown("""
    **آزمون امتیازی — نسبت دو نمونه‌ای:**
    آیا نسبت خانه‌های **بازسازی‌شده** بین خانه‌های دارای آسانسور و بدون آسانسور
    تفاوت معناداری دارد؟
    """)

    from statsmodels.stats.proportion import proportions_ztest
    import plotly.graph_objects as go

    required = ["has_elevator", "is_rebuilt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in `divar_clean.parquet`: {missing}")
        st.stop()

    dfb = df[required].copy()
    st.caption(f"Initial rows: {len(dfb):,}")

    # boolean → 1/0, NA stays NaN
    dfb["has_elevator"] = dfb["has_elevator"].map({True: 1, False: 0})
    dfb["is_rebuilt"] = dfb["is_rebuilt"].map({True: 1, False: 0})
    dfb = dfb.dropna(subset=["has_elevator", "is_rebuilt"])
    st.caption(f"Rows after removing missing values: {len(dfb):,}")

    elevator_success = int(((dfb["has_elevator"] == 1) & (dfb["is_rebuilt"] == 1)).sum())
    no_elevator_success = int(((dfb["has_elevator"] == 0) & (dfb["is_rebuilt"] == 1)).sum())
    n_elevator = int((dfb["has_elevator"] == 1).sum())
    n_no_elevator = int((dfb["has_elevator"] == 0).sum())

    if n_elevator == 0 or n_no_elevator == 0:
        st.warning("One of the groups is empty — cannot run the test.")
        st.stop()

    p_elevator = elevator_success / n_elevator
    p_no_elevator = no_elevator_success / n_no_elevator

    c1, c2 = st.columns(2)
    c1.metric("With elevator",
              f"{p_elevator * 100:.1f}%",
              f"n={n_elevator:,}, rebuilt={elevator_success:,}")
    c2.metric("Without elevator",
              f"{p_no_elevator * 100:.1f}%",
              f"n={n_no_elevator:,}, rebuilt={no_elevator_success:,}")

    cond_ok = (
        n_elevator * p_elevator >= 5
        and n_elevator * (1 - p_elevator) >= 5
        and n_no_elevator * p_no_elevator >= 5
        and n_no_elevator * (1 - p_no_elevator) >= 5
    )

    st.write(f"- with elevator:    n·p = {n_elevator * p_elevator:.1f}, "
             f"n·(1-p) = {n_elevator * (1 - p_elevator):.1f}")
    st.write(f"- without elevator: n·p = {n_no_elevator * p_no_elevator:.1f}, "
             f"n·(1-p) = {n_no_elevator * (1 - p_no_elevator):.1f}")

    if cond_ok:
        z_stat, p_value = proportions_ztest(
            count=[elevator_success, no_elevator_success],
            nobs=[n_elevator, n_no_elevator],
            alternative="two-sided",
        )
        st.write(f"**Two-sample proportion Z-test:** z = {z_stat:.3f}, p = {p_value:.3e}")
        if p_value < 0.05:
            st.success("Reject H₀ — rebuilt proportion differs significantly between the two groups.")
        else:
            st.info("Fail to reject H₀ — no significant difference.")
    else:
        st.warning("Normal-approximation condition not met.")

    fig = go.Figure(data=[go.Bar(
        x=["No Elevator", "With Elevator"],
        y=[p_no_elevator * 100, p_elevator * 100],
        text=[f"{p_no_elevator * 100:.1f}%", f"{p_elevator * 100:.1f}%"],
        textposition="outside",
    )])
    fig.update_layout(
        title="Rebuilt Proportion by Elevator Status",
        yaxis_title="Rebuilt Proportion (%)",
        yaxis_range=[0, 50],
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# Bonus #2 — Kruskal-Wallis: Building direction & price
# ==================================================================
with tabs[5]:
    st.markdown("""
    **آزمون امتیازی — Kruskal-Wallis:**
    آیا میانه‌ی قیمت بین **جهت‌های مختلف ساختمان** (شمال، جنوب، شرق، غرب)
    تفاوت معناداری دارد؟
    """)

    from scipy.stats import kruskal
    import plotly.express as px

    required = ["price_value", "building_direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in `divar_clean.parquet`: {missing}")
        st.stop()

    dfk = df[required].copy()
    st.caption(f"Initial rows: {len(dfk):,}")

    dfk = dfk.dropna(subset=["price_value", "building_direction"])
    dfk["price_value"] = pd.to_numeric(dfk["price_value"], errors="coerce")
    dfk = dfk.dropna(subset=["price_value"])
    st.caption(f"After removing missing values: {len(dfk):,}")

    Q1 = dfk["price_value"].quantile(0.25)
    Q3 = dfk["price_value"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    dfk["price_value"] = dfk["price_value"].clip(lower=lower, upper=upper)
    st.caption(f"Rows after handling outliers: {len(dfk):,}")

    st.markdown("**Sample count per direction**")
    st.dataframe(
        dfk["building_direction"].value_counts().rename_axis("direction").to_frame("count")
    )

    groups = []
    for direction in dfk["building_direction"].unique():
        groups.append(dfk.loc[dfk["building_direction"] == direction, "price_value"])

    if len(groups) < 2:
        st.warning("Need at least two groups to run Kruskal-Wallis.")
        st.stop()

    stat, p_value = kruskal(*groups)
    st.write(f"**Kruskal-Wallis:** H = {stat:.2f}, p = {p_value:.3e}")
    if p_value < 0.05:
        st.success("Reject H₀ — significant difference between the groups.")
    else:
        st.info("Fail to reject H₀ — no significant difference.")

    summary = (
        dfk.groupby("building_direction")["price_value"]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .sort_values("median", ascending=False)
    )
    st.markdown("**Price summary by building direction**")
    st.dataframe(summary.round(0), use_container_width=True)

    fig = px.bar(
        summary.reset_index(),
        x="building_direction",
        y="median",
        text="median",
        title="Median Price by Building Direction",
        labels={"median": "Median Price Value",
                "building_direction": "Building Direction"},
    )
    fig.update_traces(texttemplate="%{text:.3s}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# Bonus #3 — Chi-Square: Property type & deed type
# ==================================================================
with tabs[6]:
    st.markdown("""
    **آزمون امتیازی — کای دو استقلال:**
    آیا **نوع ملک** (`property_type`) و **نوع سند** (`deed_type`) به هم وابسته‌اند؟
    """)

    from scipy.stats import chi2_contingency
    import plotly.express as px

    required = ["property_type", "deed_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in `divar_clean.parquet`: {missing}")
        st.stop()

    dfc = df[required].copy()
    st.caption(f"Initial rows: {len(dfc):,}")

    dfc = dfc.dropna(subset=["property_type", "deed_type"])
    st.caption(f"Rows after removing missing values: {len(dfc):,}")

    contingency_table = pd.crosstab(dfc["property_type"], dfc["deed_type"])
    st.markdown("**Contingency table (observed frequencies)**")
    st.dataframe(contingency_table, use_container_width=True)

    if contingency_table.size == 0:
        st.warning("Empty contingency table — cannot run the test.")
        st.stop()

    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    cells_lt5 = int((expected < 5).sum())
    total_cells = expected.size
    pct_lt5 = cells_lt5 / total_cells * 100

    st.write(f"- Total cells: {total_cells}")
    st.write(f"- Cells with expected count < 5: {cells_lt5} ({pct_lt5:.1f}%)")

    if pct_lt5 <= 20 and expected.min() > 0:
        st.write(f"**Chi-Square:** χ² = {chi2_stat:.2f}, dof = {dof}, p = {p_value:.3e}")
        if p_value < 0.05:
            st.success("Reject H₀ — property_type and deed_type are **dependent**.")
        else:
            st.info("Fail to reject H₀ — independent.")
    else:
        st.warning("Chi-Square conditions not met.")

    summary = (
        pd.crosstab(dfc["property_type"], dfc["deed_type"], normalize="index") * 100
    )
    st.markdown("**Deed type % distribution by property type**")
    st.dataframe(summary.round(2), use_container_width=True)

    plot_df = summary.reset_index().melt(
        id_vars="property_type",
        var_name="deed_type",
        value_name="pct",
    )
    fig = px.bar(
        plot_df,
        x="property_type",
        y="pct",
        color="deed_type",
        title="Deed Type Proportion by Property Type",
        labels={"pct": "Proportion (%)", "property_type": "Property Type"},
    )
    fig.update_layout(barmode="stack", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)