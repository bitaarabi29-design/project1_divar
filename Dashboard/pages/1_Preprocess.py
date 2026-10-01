import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_utils import load_data

st.set_page_config(page_title="Preprocess", layout="wide")
st.title("🧹 Preprocess — Data Quality Report")
st.caption("Output of `Final_Preprocess.ipynb`. This page documents what the file contains.")

df = load_data()
st.success(f"Clean dataset: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")

tabs = st.tabs(["📊 Overview", "🕳️ Missing Values", "📖 Column Guide"])

# ==================================================================
# Tab 1 — Overview
# ==================================================================
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    col1, col2 = st.columns(2)
    with col1:
        dtypes = df.dtypes.astype(str).value_counts().reset_index()
        dtypes.columns = ['dtype', 'count']
        st.plotly_chart(px.bar(dtypes, x='dtype', y='count',
                               title="Columns by dtype"),
                        use_container_width=True)
    with col2:
        d = df['deal_type'].value_counts().reset_index()
        d.columns = ['deal_type', 'count']
        st.plotly_chart(px.pie(d, names='deal_type', values='count',
                               hole=0.5, title="Deal type distribution"),
                        use_container_width=True)

# ==================================================================
# Tab 2 — Missing Values
# ==================================================================
with tabs[1]:
    sale = df['deal_type'] == 'sale'
    rent = df['deal_type'] == 'rent'

    cleared = pd.DataFrame({
        'coordinates':          df['location_issue'].value_counts(),
        'building size':        df['building_size_issue'].value_counts(),
        'land size':            df['land_size_issue'].value_counts(),
        'sale price':           df.loc[sale, 'price_issue'].value_counts(),
        'rent (deposit, rent)': df.loc[rent, 'price_issue'].value_counts(),
        'nightly prices':       df.loc[df['deal_type'] == 'daily_rent', 'price_issue'].value_counts(),
    }).T.fillna(0).astype(int)

    fig = go.Figure()
    for col in cleared.columns:
        fig.add_bar(y=cleared.index, x=cleared[col], name=col, orientation='h')
    fig.update_layout(barmode='stack', height=420,
                      xaxis_title="Number of ads",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    na = df.isna().mean().sort_values(ascending=False).head(20).reset_index()
    na.columns = ['column', 'missing_pct']
    na['missing_pct'] *= 100
    st.plotly_chart(
        px.bar(na, x='missing_pct', y='column', orientation='h',
               labels={'missing_pct': 'Missing (%)'},
               title="Top 20 columns by missing share")
          .update_yaxes(categoryorder='total ascending')
          .update_layout(height=520),
        use_container_width=True,
    )

# ==================================================================
# Tab 3 — Column Guide
# ==================================================================
with tabs[2]:
    st.markdown("""
### How the file works

Every cleaned column keeps its **original name**. Any value that was
cleared becomes `NaN`, and the reason goes into a matching `*_issue` column.

---

#### What was added

| Group | Columns |
|---|---|
| **Labels** | `deal_type` (sale / rent / daily_rent / service), `is_residential_unit`, `is_metro`, `in_main_window`, `jalali_month`, `is_wanted_ad`, `is_partial_offer` |
| **Derived prices** | `unified_price` (sale price, or `credit + rent/0.03` for rent), `price_per_m2` |
| **Coordinates** | `utm_x`, `utm_y` (UTM zone 39N, in meters) |
| **Combined amenities** | `has_X_combined` for balcony, pool, sauna, jacuzzi, barbecue, security_guard — form field **OR** text mention |
| **Keyword features** | `mentions_X` for new building, luxury, lobby, loan, installments, presale, mehr housing, key money, agency, etc. |
| **Quality flags** | `location_issue`, `building_size_issue`, `land_size_issue`, `price_issue`, `far_from_city` |

---

#### What changed meaning

| Column | Reads as |
|---|---|
| `construction_year` | `1370` means **1370 or earlier** |
| `floor`, `total_floors_count`, `extra_person_capacity` | `30` means **30 or more** |
| `unit_per_floor` | `9` means **9 or more** |
| `rooms_count` | `5` means **5 or more** |
| `building_size` in `plot-old` | holds **land area**, not built area |

---

#### What was dropped

- `rent_to_single`, `location_radius` — almost empty or unrelated to the map pin
- `transformable_credit`, `transformable_rent` — duplicate of `credit_value` / `rent_value`
- `transformed_credit`, `transformed_rent` — intermediate calc columns
- 1 ad without a category + 2,590 duplicate ads

Raw: 1,000,000 × 60 → Clean: 997,409 × 88.

---

#### Quality flags

| Column | Values |
|---|---|
| `location_issue` | `outside_iran`, `shared_pin` |
| `building_size_issue` / `land_size_issue` | `placeholder`, `implausible` |
| `price_issue` | `placeholder`, `zero`, `implausible` |
| `far_from_city` | `True` when the pin is >50 km from the city median (coordinates kept) |

---

#### How to use it

| Goal | Filter |
|---|---|
| Price stats / models | one `deal_type` at a time, `is_wanted_ad == False`, `is_partial_offer == False` |
| Monthly trends | `in_main_window == True` |
| Size hypotheses | `is_residential_unit == True` |
| Amenity studies | use the `has_X_combined` columns |
""")