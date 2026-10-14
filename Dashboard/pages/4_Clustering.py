import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Clustering", layout="wide")

ASSETS = Path(__file__).parent.parent / "assets" / "clustering_plots"

EXPECTED_PLOTS = [
    "kmeans_scatter_k10.png",
    "kmeans_elbow.png",
    "macro_regions_map.png",
    "macro_regions_bar.png",
    "macro_regions_box.png",
    "macro_regions_violin.png",
    "macro_regions_hexbin.png",
]

KMEANS_FEATURES = [
    "location_latitude",
    "location_longitude",
    "price_value",
    "building_size",
    "rooms_count",
]


def show_image(filename: str, caption: str = "", width: int = 650) -> None:
    """Render a pre-rendered PNG at a fixed width, or a neutral notice."""
    path = ASSETS / filename
    if path.exists():
        st.image(str(path), caption=caption or None, width=width)
    else:
        st.info(
            f"Image `{filename}` is not available under "
            f"`assets/clustering_plots/`."
        )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🔵 Clustering — Divar Real-Estate Segmentation")
st.markdown(
    "This page presents the output of two clustering algorithms — **KMeans** "
    "and **DBSCAN** — applied to the Divar real-estate listings. Every chart "
    "and table below was produced offline in the notebooks; this page only "
    "renders the results."
)

with st.expander("Expected files in `assets/clustering_plots/`", expanded=False):
    for fname in EXPECTED_PLOTS:
        exists = "✅" if (ASSETS / fname).exists() else "❌"
        st.markdown(f"{exists} `{fname}`")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Preparation",
    "KMeans Clustering",
    "Elbow Method",
    "DBSCAN",
    "Three Macro-Regions",
])


# ---------------------------------------------------------------------------
# Tab 1 — Data preparation
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Feature set used for clustering")

    st.markdown(
        "The KMeans features below were selected from the cleaned Divar "
        "parquet. Rows with missing values in these columns were dropped "
        "before fitting. `rooms_count` was originally stored as text and "
        "was converted to a numeric scale (`بدون اتاق` → 0, "
        "`پنج یا بیشتر` → 5). Latitude/longitude were converted to UTM "
        "using `utm.from_latlon(..., force_zone_number=39)`."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in source parquet", "997,409")
    c2.metric("Rows after dropna", "277,711")
    c3.metric("Features used", len(KMEANS_FEATURES))

    st.markdown("**Features:**")
    st.code("\n".join(KMEANS_FEATURES), language="text")


# ---------------------------------------------------------------------------
# Tab 2 — KMeans clustering
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("KMeans with k = 10")

    st.markdown(
        "Features are standardised with `StandardScaler`. KMeans is then fit "
        "with **k = 10**, **n_init = 10**, **random_state = 44**. The ten "
        "cluster centroids are inverse-transformed back to the original "
        "feature scale and projected into UTM so they can be overlaid on the "
        "scatter plot below."
    )

    show_image(
        "kmeans_scatter_k10.png",
        "Property clusters in UTM coordinates (k = 10). Black × marks are "
        "cluster centroids.",
    )

    st.markdown("**Cluster sizes:**")
    sizes = (
        pd.DataFrame(
            {
                "cluster": list(range(10)),
                "listings": [94143, 21965, 38308, 2, 330, 39961, 1, 20, 57414, 25567],
            }
        )
        .set_index("cluster")
        .assign(share=lambda x: (x["listings"] / x["listings"].sum()).map("{:.1%}".format))
    )
    st.dataframe(sizes, use_container_width=True)

    st.markdown(
        "Clusters 3, 6 and 7 are effectively outliers — they contain only 2, "
        "1 and 20 listings respectively. The bulk of the data is split across "
        "clusters 0, 8, 5 and 2."
    )


# ---------------------------------------------------------------------------
# Tab 3 — Elbow + metrics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Elbow curve and cluster-quality metrics")

    st.markdown(
        "WCSS (within-cluster sum of squares) is computed for `K` from 1 to "
        "10. Silhouette, Davies-Bouldin and Calinski-Harabasz scores are "
        "computed for `K` from 2 to 10. Silhouette uses `sample_size=10000` "
        "for speed."
    )

    show_image("kmeans_elbow.png", width=600)

    st.markdown("**Metrics by K:**")
    metrics = pd.DataFrame(
        {
            "K": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Silhouette": [0.4155, 0.4172, 0.4172, 0.4711, 0.4065,
                           0.4065, 0.4121, 0.4316, 0.4394],
            "Davies-Bouldin": [1.1529, 1.0176, 0.7210, 0.6811, 0.6955,
                               0.6111, 0.6718, 0.6396, 0.6299],
            "Calinski-Harabasz": [50960.7, 55443.1, 61878.7, 74719.0, 85276.5,
                                  88598.4, 93987.4, 94403.5, 97249.9],
        }
    ).set_index("K")
    st.dataframe(metrics.round(4), use_container_width=True)

    st.info("The elbow curve suggests **K = 6** as a reasonable cut-off.")


# ---------------------------------------------------------------------------
# Tab 4 — DBSCAN
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("DBSCAN on geographic features")

    st.markdown(
        "DBSCAN was run on three standardised features: `utm_x_km`, "
        "`utm_y_km` and `log_price`. Because the full 591,395-row fit took "
        "roughly 450 seconds per run on a GPU (via cuML), the run was not "
        "reproduced live here — only its conclusions are reported."
    )

    st.markdown(
        """
**What the notebook found:**

- With small `eps` and low `min_samples`, DBSCAN collapses almost all
  listings into **one giant cluster** — with `eps=0.60, min_samples=50`,
  **591,020 of 591,395** points land in a single cluster.
- Raising `min_samples` forces DBSCAN to treat sparser areas as noise. With
  **`eps=0.35, min_samples=2000`** the result is **6 clusters and ≈ 9.4 %**
  noise — a much more meaningful segmentation.
- Re-running DBSCAN on the giant Tehran cluster alone with
  `eps=0.20, min_samples=2000` splits it into four sub-clusters that map
  cleanly onto Tehran, Isfahan, Rasht and Tabriz.
- The final step fills the remaining noise points with a KNN classifier
  fitted on the labelled points, so no listings are discarded.
        """
    )

    st.markdown("**Selected runs:**")
    runs = pd.DataFrame(
        {
            "eps": [0.35, 0.35, 0.60],
            "min_samples": [50, 2000, 50],
            "clusters": [8, 6, 3],
            "noise": ["0.3 %", "9.4 %", "0.0 %"],
            "notebook runtime": ["383 s", "328 s", "471 s"],
        }
    )
    st.dataframe(runs, use_container_width=True)

    st.markdown(
        "The map in the next tab shows the final geographic layout after the "
        "noise-fill step and the manual mapping of intermediate clusters to "
        "macro-regions."
    )


# ---------------------------------------------------------------------------
# Tab 5 — Three macro-regions
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Nine intermediate clusters → three macro-regions")

    st.markdown(
        "The six outer DBSCAN clusters, plus three sub-clusters obtained by "
        "re-running DBSCAN on the largest (Tehran) cluster, were manually "
        "grouped into **three macro-regions** so the segmentation is easy to "
        "interpret in downstream price analysis."
    )

    show_image(
        "macro_regions_map.png",
        "Macro-regions across Iran (30,000-listing sample).",
        width=700,
    )

    summary = pd.DataFrame(
        {
            "listings": [383792, 107540, 44477],
            "median_price_rial": [1.800e9, 1.417e9, 1.150e9],
            "mean_price_rial": [4.494e9, 2.550e9, 2.418e9],
            "mean_x_km": [511, 621, 568],
            "mean_y_km": [3917, 4117, 3340],
            "top_3_cities": [
                "tehran, karaj, isfahan",
                "mashhad, tabriz, rasht",
                "shiraz, ahvaz, kerman",
            ],
        },
        index=[
            "Central (Tehran + Isfahan)",
            "North (Caspian + NW + NE)",
            "South (Fars + Khuzestan + Kerman)",
        ],
    )

    st.markdown("**Region summary:**")
    st.dataframe(summary, use_container_width=True)

    st.markdown("#### Listings per region")
    show_image("macro_regions_bar.png", width=600)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Price distribution — box")
        show_image("macro_regions_box.png", width=500)
    with c2:
        st.markdown("#### Price distribution — violin")
        show_image("macro_regions_violin.png", width=500)

    st.markdown("#### Median price per cell")
    show_image(
        "macro_regions_hexbin.png",
        "Median log10 price per UTM hex cell, split by macro-region.",
        width=900,
    )

    st.info(
        "The Central region holds ~70 % of all listings and the highest "
        "median price. The North and South regions are visibly separated in "
        "both volume and price level."
    )