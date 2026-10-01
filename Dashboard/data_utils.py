import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "divar_clean.parquet"

@st.cache_data(show_spinner="Loading cleaned Divar dataset…")
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)