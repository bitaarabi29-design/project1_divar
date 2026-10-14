import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Divar Real-Estate Dashboard",
                   layout="wide", page_icon="🏠")

col1, col2 = st.columns([1, 3], vertical_alignment="center")
logo_path = Path("assets/logo.png")
with col1:
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.info("Put your logo in `assets/logo.png`")
with col2:
    st.title("Divar Real-Estate Ads — Analysis & Prediction")
    st.caption("Quera AI Bootcamp — Summer 1405 — Project 1")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("👥 Team Members")
    st.write("- Bita Arabi")
    st.write("- Pardis Eslami")
    st.write("- Behzad Dadmehr")
    st.write("- Aria Hosseini")
with c2:
    st.subheader("🎓 Mentor")
    st.write("Melika Molaei")

st.divider()
st.subheader("📊 Navigate the dashboard")
c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("pages/1_Preprocess.py",       label="🧹 Preprocess",        icon="➡️")
    st.page_link("pages/2_EDA.py",              label="📈 EDA & Descriptive", icon="➡️")
with c2:
    st.page_link("pages/3_Hypothesis_Tests.py", label="🧪 Hypothesis Tests",  icon="➡️")
    st.page_link("pages/4_Clustering.py",   label="🎯 Clustering",    icon="➡️")
with c3:
    st.page_link("pages/5_Regression.py",       label="📉 Regression",        icon="➡️")