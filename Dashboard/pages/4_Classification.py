import streamlit as st
from data_utils import load_data
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

st.set_page_config(page_title="Classification & Clustering", layout="wide")
st.title("Classification & Clustering")


# -----------------------------------------------------------------------------
# Plot folders (nested inside assets/classification_plots/)
# -----------------------------------------------------------------------------
PLOTS_ROOT = Path(__file__).parent.parent / "assets" / "classification_plots"
CLUSTERS_DIR = PLOTS_ROOT / "clusters"
DBSCAN_DIR = PLOTS_ROOT / "dbscan"

# Smaller display size for stored images (avoids huge full-width charts)
IMG_WIDTH = 620


def _img(path: Path, caption: str = None, width: int = IMG_WIDTH) -> bool:
    """Display a stored PNG if it exists, otherwise show an info box."""
    if path.exists():
        st.image(str(path), width=width, caption=caption or path.name)
        return True
    st.info(f"Plot not found: {path.name}")
    return False


def _img_pair(path_left: Path, cap_left: str, path_right: Path, cap_right: str) -> None:
    """Show two images side by side in two columns."""
    c1, c2 = st.columns(2)
    with c1:
        _img(path_left, cap_left, width=IMG_WIDTH)
    with c2:
        _img(path_right, cap_right, width=IMG_WIDTH)


# -----------------------------------------------------------------------------
# Data preparation (only used as a live fallback when an image is missing)
# -----------------------------------------------------------------------------
ROOMS_MAP = {
    "بدون اتاق": 0,
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج یا بیشتر": 5,
}

PRICE_CANDIDATES = ["target_price", "unified_price", "price_value"]


def _pick_col(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner="Preparing clustering data…")
def _prepare_data(sample_n=60000, random_state=44):
    df = load_data().copy()

    price_col = _pick_col(df, PRICE_CANDIDATES)
    if price_col is None:
        return None

    required = ["utm_x", "utm_y", "building_size", "rooms_count", price_col]
    for c in required:
        if c not in df.columns:
            return None

    if df["rooms_count"].dtype == object:
        df["rooms_count"] = df["rooms_count"].map(ROOMS_MAP)

    mask = (
        df["utm_x"].notna()
        & df["utm_y"].notna()
        & df[price_col].notna()
        & df["building_size"].notna()
        & df["rooms_count"].notna()
    )

    if "is_duplicate" in df.columns:
        mask &= ~df["is_duplicate"].fillna(False).astype(bool)

    if "price_is_outlier" in df.columns:
        mask &= ~df["price_is_outlier"].fillna(False).astype(bool)

    data = df.loc[mask].copy()
    data["log_price"] = np.log10(data[price_col].clip(lower=1))
    data["price_for_cluster"] = data[price_col]

    features = ["utm_x", "utm_y", "price_for_cluster", "building_size", "rooms_count"]
    data = data.dropna(subset=features)

    if len(data) > sample_n:
        data = data.sample(sample_n, random_state=random_state)

    X = data[features].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[X.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return data, X_scaled, scaler, features, price_col


@st.cache_data(show_spinner="Computing WCSS…")
def _compute_wcss(k_max=20, sample_n=25000, random_state=44):
    out = _prepare_data(sample_n=sample_n, random_state=random_state)
    if out is None:
        return None

    _, X_scaled, _, _, _ = out
    ks = list(range(1, k_max + 1))
    wcss = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        wcss.append(float(km.inertia_))
    return ks, wcss


@st.cache_data(show_spinner="Computing cluster metrics…")
def _compute_metrics(k=10, sample_n=15000, random_state=44):
    out = _prepare_data(sample_n=sample_n, random_state=random_state)
    if out is None:
        return None

    _, X_scaled, _, _, _ = out
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_scaled)

    return {
        "Silhouette Score": float(silhouette_score(X_scaled, labels)),
        "Davies-Bouldin Index": float(davies_bouldin_score(X_scaled, labels)),
        "Calinski-Harabasz Index": float(calinski_harabasz_score(X_scaled, labels)),
    }


# -----------------------------------------------------------------------------
# Page body
# -----------------------------------------------------------------------------
with st.expander("All stored plots in assets/classification_plots"):
    st.markdown("**clusters/**")
    if CLUSTERS_DIR.exists():
        for p in sorted(CLUSTERS_DIR.glob("*.png")):
            _img(p, width=IMG_WIDTH)
    else:
        st.info("clusters/ folder not found.")

    st.markdown("**dbscan/**")
    if DBSCAN_DIR.exists():
        for p in sorted(DBSCAN_DIR.glob("*.png")):
            _img(p, width=IMG_WIDTH)
    else:
        st.info("dbscan/ folder not found.")


tabs = st.tabs(
    [
        "1) K-Means (k=10)",
        "2) Elbow Method",
        "3) Cluster Evaluation",
        "4) DBSCAN",
        "5) Macro Regions",
    ]
)


# -----------------------------------------------------------------------------
# Tab 1 — K-Means k=10
# -----------------------------------------------------------------------------
with tabs[0]:
    # Course requirement: Persian question statement at the top of each tab
    st.markdown(
        "**خوشه‌بندی املاک با K-Means با k=10 روی مختصات UTM و قیمت، همراه با نمایش مراکز خوشه‌ها**"
    )

    st.markdown(
        """
        In `clustering.ipynb`, the data is first cleaned by dropping rows with
        missing values in the key columns — `location_latitude`, `location_longitude`,
        `price_value`, `building_size`, and `rooms_count`. The `rooms_count`
        column is stored as Persian text in the raw data and is mapped to
        integers (`بدون اتاق` → 0, `پنج یا بیشتر` → 5).

        The feature set used for K-Means is:

        - `location_latitude`, `location_longitude` — geographic coordinates
        - `price_value` — listing price
        - `building_size` — property size
        - `rooms_count` — number of rooms

        These features are scaled with `StandardScaler`, and then
        **KMeans(n_clusters=10, n_init=10, random_state=44)** is fitted.
        Each property receives a `cluster` label from 0 to 9, and the ten
        cluster centroids are drawn on the UTM scatter plot as black `X`
        markers. Geographic coordinates are projected to UTM zone 39N
        (`utm_x`, `utm_y`) so the plot is in meters.
        """
    )

    shown = _img(
        CLUSTERS_DIR / "PropertyCluster.png",
        caption="Property clusters with k=10 — centroids shown as black X markers.",
    )
    if not shown:
        st.warning(
            "PropertyCluster.png was not found. Please place it in "
            "assets/classification_plots/clusters/."
        )


# -----------------------------------------------------------------------------
# Tab 2 — Elbow
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown(
        "**انتخاب تعداد بهینه‌ی خوشه با روش آرنج — مقدار WCSS برای مقادیر مختلف k**"
    )

    st.markdown(
        """
        The notebook computes **WCSS** (Within-Cluster Sum of Squares, i.e. the
        inertia of K-Means) for two different ranges of `k`:

        - once for `k = 1 … 20`
        - once for `k = 1 … 10`

        Both curves are shown below. Looking at the elbow — where the slope
        of the curve flattens — and combining that with the evaluation metrics
        (Silhouette, Davies-Bouldin, and Calinski-Harabasz, shown in the next
        tab), the value **k = 10** was chosen as the number of clusters used
        for the final model.
        """
    )

    elbow_1 = CLUSTERS_DIR / "ElbowMethod.png"
    elbow_2 = CLUSTERS_DIR / "ElbowMethod2.png"

    if elbow_1.exists() or elbow_2.exists():
        # At least one stored chart is available — show whichever exists.
        _img_pair(
            elbow_1,
            "Elbow method for k = 1 … 20",
            elbow_2,
            "Elbow method for k = 1 … 10",
        )
    else:
        # No stored charts — compute a live WCSS curve as a fallback.
        wcss_out = _compute_wcss(k_max=20, sample_n=25000)
        if wcss_out is None:
            st.warning("Could not compute WCSS from the data.")
        else:
            import plotly.express as px

            ks, wcss = wcss_out
            fig = px.line(
                x=ks,
                y=wcss,
                markers=True,
                labels={"x": "Number of clusters (k)", "y": "WCSS"},
                title="Elbow method — live WCSS computation",
            )
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 3 — Metrics
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown(
        "**ارزیابی کیفیت خوشه‌ها با معیارهای Silhouette، Davies-Bouldin و Calinski-Harabasz**"
    )

    st.markdown(
        """
        Three standard clustering metrics are reported in the notebook for
        the 10-cluster solution:

        - **Silhouette Score** — closer to 1 means better-separated clusters.
          Value at k=10: ≈ **0.4885**
        - **Davies-Bouldin Index** — lower is better; it reflects high
          within-cluster cohesion and low between-cluster similarity.
          Value at k=10: ≈ **0.7285**
        - **Calinski-Harabasz Index** — higher is better.
          Value at k=10: ≈ **133710.7**

        These three metrics were also compared across `k = 2 … 10` in the
        notebook to confirm that `k = 10` gives a good trade-off between
        quality and the number of clusters.
        """
    )

    st.markdown("**Live-computed metrics on a subsample:**")

    metrics = _compute_metrics(k=10, sample_n=15000)

    if metrics is None:
        st.warning("Could not compute clustering metrics.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{metrics['Silhouette Score']:.4f}")
        c2.metric("Davies-Bouldin Index", f"{metrics['Davies-Bouldin Index']:.4f}")
        c3.metric("Calinski-Harabasz Index", f"{metrics['Calinski-Harabasz Index']:.1f}")

        st.caption(
            "These metrics are computed on a smaller subsample so the page stays "
            "responsive. The full-data values reported in the notebook are shown "
            "in the text above."
        )


# -----------------------------------------------------------------------------
# Tab 4 — DBSCAN
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown(
        "**خوشه‌بندی DBSCAN روی مختصات UTM و قیمت و تنظیم پارامترهای eps و min_samples**"
    )

    st.markdown(
        """
        In `DBSCAN.ipynb`, **cuML DBSCAN** is run on the **GPU** over roughly
        **591,395 records**. The input features are:

        - `utm_x_km`, `utm_y_km` — UTM coordinates in kilometers
        - `log_price` — log10 of the listing price

        The data is scaled with `StandardScaler` before running DBSCAN. To
        avoid re-uploading the whole array on every fit, the input is moved
        to the GPU once with `cp.asarray(...)`.

        #### Parameter tuning

        The first attempts used `min_samples=50`. With that setting, DBSCAN
        collapsed almost the entire dataset into a **single giant cluster**
        with essentially **0% noise** — it could not separate dense regions.

        Increasing `min_samples` gave a much more meaningful result:

        - `eps=0.35`, `min_samples=2000` → **6 clusters**, about **9.4% noise**
        - The biggest cluster was then re-clustered with
          `eps=0.20`, `min_samples=2000`, producing **4 sub-clusters**
          (about 20.7% noise inside that subset).

        #### Mapping to 3 macro-regions

        The nine intermediate clusters (6 outer + 4 sub, minus overlap) were
        mapped by hand to **three macro-regions**:

        - **Central** — Tehran + Isfahan
        - **North** — Caspian provinces + NW + NE
        - **South** — Fars + Khuzestan + Kerman + Hormozgan

        #### Filling the noise with KNN

        To keep the noise points in the final dataset, a
        `KNeighborsClassifier(n_neighbors=5, weights='distance')` was trained
        on the labeled points and used to assign each noise point to its
        nearest sub-cluster. After this step the final noise fraction dropped
        to about **9.4%**.

        *(The final macro-region map is displayed in the "5) Macro Regions"
        tab.)*
        """
    )


# -----------------------------------------------------------------------------
# Tab 5 — Macro regions (map + distributions)
# -----------------------------------------------------------------------------
with tabs[4]:
    st.markdown(
        "**مناطق کلان مسکن — نقشه، حجم داده، و توزیع قیمت در هر منطقه**"
    )

    st.markdown(
        """
        The output of the two-stage DBSCAN + manual mapping is shown here.
        Each macro-region is drawn in a different color on the map, and its
        price distribution is compared against the others using box and
        violin plots.

        The **Barplot** shows the number of listings per macro-region —
        the Central region dominates by an order of magnitude, which is
        expected given that it contains Tehran and Karaj.
        """
    )

    # 1) The map of macro-regions
    _img(
        DBSCAN_DIR / "Map.png",
        caption="Macro-regions of Iran derived from DBSCAN on UTM coordinates and price.",
    )

    # 2) Barplot of listing counts per region
    _img(
        DBSCAN_DIR / "Barplot.png",
        caption="Number of listings per macro-region — the Central region dominates.",
    )

    # 3) Box and Violin side by side
    _img_pair(
        DBSCAN_DIR / "Boxplot.png",
        "Price distribution per region (box plot).",
        DBSCAN_DIR / "ViolinPlot.png",
        "Price distribution per region (violin plot).",
    )

    # 4) Price-only box plot
    _img(
        DBSCAN_DIR / "BoxPlotPrice.png",
        caption="Median price per macro-region — Central is the most expensive, South the cheapest.",
    )

    st.markdown("#### Macro-region summary table")

    macro_summary = pd.DataFrame(
        [
            {
                "Region": "Central (Tehran + Isfahan)",
                "Listings": 383792,
                "Median price (rial)": 1.80e9,
                "Mean x (km)": 511,
                "Mean y (km)": 3917,
                "Top 3 cities": "tehran (132,912), karaj (33,723), isfahan (21,617)",
            },
            {
                "Region": "North (Caspian + NW + NE)",
                "Listings": 107540,
                "Median price (rial)": 1.416667e9,
                "Mean x (km)": 621,
                "Mean y (km)": 4117,
                "Top 3 cities": "mashhad (33,244), tabriz (14,598), rasht (10,802)",
            },
            {
                "Region": "South (Fars + Khuzestan + Kerman)",
                "Listings": 44477,
                "Median price (rial)": 1.15e9,
                "Mean x (km)": 568,
                "Mean y (km)": 3340,
                "Top 3 cities": "shiraz (16,699), ahvaz (7,604), kerman (2,432)",
            },
        ]
    )

    st.dataframe(macro_summary, use_container_width=True)

    st.markdown(
        """
        **Reading the three regions:**

        - **Central** — largest share of listings and the highest median price,
          driven by Tehran and Karaj.
        - **North** — mid-range median price with the widest spread, since it
          mixes very different cities (Mashhad, Tabriz, Rasht, …).
        - **South** — smallest share of listings and the lowest median price,
          covering Shiraz, Ahvaz, and Kerman.
        """
    )


st.markdown("---")
st.caption(
    "This page displays the K-Means, Elbow, evaluation-metric, and DBSCAN-derived "
    "macro-region outputs. All plots are read from assets/classification_plots."
)