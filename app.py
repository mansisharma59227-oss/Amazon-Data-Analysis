"""Amazon Product Analysis Dashboard (Streamlit) — built from amazoneda.ipynb"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------- Amazon theme ----------
ORANGE, NAVY, DARK, LIGHT, TEAL = "#FF9900", "#232F3E", "#131A22", "#EAEDED", "#37475A"
PALETTE = [ORANGE, "#146EB4", "#00A8E1", "#FEBD69", "#879596", "#F3A847", "#067D62", "#C7511F"]
ICON_PATH = Path(__file__).parent / "assets" / "amazon_icon.png"

st.set_page_config(
    page_title="Amazon Product Analysis",
    page_icon=Image.open(ICON_PATH) if ICON_PATH.exists() else "🛒",
    layout="wide",
)

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 1rem; }}
  header[data-testid="stHeader"] {{ background: {DARK}; }}
  .amz-header {{ background:{NAVY}; border-bottom:4px solid {ORANGE}; border-radius:8px;
                 padding:14px 22px; display:flex; align-items:center; gap:18px; margin-bottom:18px; }}
  .amz-header h1 {{ color:#fff; font-size:1.5rem; margin:0; font-weight:600; }}
  .amz-header span {{ color:{ORANGE}; font-size:.9rem; }}
  .kpi {{ background:{NAVY}; border-left:5px solid {ORANGE}; border-radius:6px; padding:12px 16px; }}
  .kpi .l {{ color:#B6C0C8; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }}
  .kpi .v {{ color:#fff; font-size:1.6rem; font-weight:700; }}
  button[data-baseweb="tab"][aria-selected="true"] {{ color:{ORANGE} !important; }}
  div[data-baseweb="tab-highlight"] {{ background-color:{ORANGE} !important; }}
  section[data-testid="stSidebar"] {{ border-right:2px solid {ORANGE}; }}
</style>
<div class="amz-header">
  <svg width="130" height="44" viewBox="0 0 130 44" xmlns="http://www.w3.org/2000/svg">
    <text x="4" y="26" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#fff">amazon</text>
    <path d="M10 33 Q 58 48 104 31" stroke="{ORANGE}" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M98 25 L108 30 L97 37 Z" fill="{ORANGE}"/>
  </svg>
  <div><h1>Product Data Analysis Dashboard</h1><span>Exploratory Data Analysis</span></div>
</div>
""",
    unsafe_allow_html=True,
)


def style(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height, colorway=PALETTE,
        paper_bgcolor=NAVY, plot_bgcolor=NAVY, font_color=LIGHT,
        margin=dict(l=20, r=20, t=50, b=20), title_font_color=ORANGE,
    )
    fig.update_xaxes(gridcolor=TEAL)
    fig.update_yaxes(gridcolor=TEAL)
    return fig


# ---------- Data loading & cleaning (same steps as the notebook) ----------
@st.cache_data(show_spinner="Loading data…")
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source, encoding="utf-8")
    # numeric cleanup
    df["rating"] = pd.to_numeric(df["rating"].replace("|", np.nan), errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["rating_count"] = pd.to_numeric(
        df["rating_count"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["rating_count"] = df["rating_count"].fillna(df["rating_count"].median())
    for c in ["discounted_price", "actual_price"]:
        df[c] = (df[c].astype(str).str.replace("₹", "", regex=False)
                 .str.replace(",", "", regex=False).astype(float))
    df["discount_percentage"] = (df["discount_percentage"].astype(str)
                                 .str.replace("%", "", regex=False).astype(float))
    # engineered columns
    df["price_difference"] = df["actual_price"] - df["discounted_price"]
    df["savings_percentage"] = df["price_difference"] / df["actual_price"] * 100
    df["main_category"] = df["category"].str.split("|").str[0]
    df["sub_category"] = df["category"].str.split("|").str[1]
    return df


with st.sidebar:
    st.image(str(ICON_PATH), width=64) if ICON_PATH.exists() else None
    st.header("Data & Filters")
    up = st.file_uploader("Upload amazon.csv", type="csv")
    default = Path(__file__).parent / "amazon.csv"
    source = up if up else (default if default.exists() else None)

if source is None:
    st.info("Upload **amazon.csv** in the sidebar, or place it next to `app.py`.")
    st.stop()

df = load_data(source)

with st.sidebar:
    cats = sorted(df["main_category"].dropna().unique())
    sel_cats = st.multiselect("Main category", cats, default=cats)
    r_lo, r_hi = float(df.rating.min()), float(df.rating.max())
    rating_rng = st.slider("Rating", r_lo, r_hi, (r_lo, r_hi), 0.1)
    disc_rng = st.slider("Discount %", 0, 100, (0, 100))
    p_max = int(df.actual_price.max())
    price_rng = st.slider("Actual price (₹)", 0, p_max, (0, p_max))

f = df[
    df.main_category.isin(sel_cats)
    & df.rating.between(*rating_rng)
    & df.discount_percentage.between(*disc_rng)
    & df.actual_price.between(*price_rng)
]
if f.empty:
    st.warning("No products match these filters.")
    st.stop()

# ---------- KPIs ----------
kpis = [
    ("Products", f"{f.product_id.nunique():,}"),
    ("Avg Rating", f"{f.rating.mean():.2f} ★"),
    ("Avg Discount", f"{f.discount_percentage.mean():.1f}%"),
    ("Avg Actual Price", f"₹{f.actual_price.mean():,.0f}"),
    ("Avg Rating Count", f"{f.rating_count.mean():,.0f}"),
]
for col, (l, v) in zip(st.columns(len(kpis)), kpis):
    col.markdown(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div></div>',
                 unsafe_allow_html=True)
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["Distributions", "Categories", "Relationships", "Outliers", "Top Products"])

# ---------- Distributions ----------
with t1:
    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(f, x="rating", nbins=10, marginal="box",
                   title="Distribution of Product Ratings")), use_container_width=True)
    b.plotly_chart(style(px.histogram(f, x="actual_price", nbins=30,
                   title="Actual Price Distribution")), use_container_width=True)
    c, d = st.columns(2)
    c.plotly_chart(style(px.histogram(f, x="discount_percentage", nbins=20,
                   title="Discount Percentage Distribution")), use_container_width=True)
    d.plotly_chart(style(px.histogram(f, x="rating_count", nbins=30,
                   title="Rating Count Distribution")), use_container_width=True)

# ---------- Categories ----------
with t2:
    g = f.groupby("main_category").agg(
        products=("product_id", "count"), avg_rating=("rating", "mean"),
        avg_discount=("discount_percentage", "mean"),
        avg_actual=("actual_price", "mean"), avg_discounted=("discounted_price", "mean"),
    ).reset_index()

    def bar(col, title, y_label):
        d_ = g.sort_values(col, ascending=False)
        fig = px.bar(d_, x="main_category", y=col, title=title,
                     labels={"main_category": "Main Category", col: y_label},
                     color_discrete_sequence=[ORANGE])
        fig.update_xaxes(tickangle=-40)
        return style(fig)

    a, b = st.columns(2)
    a.plotly_chart(bar("products", "Product Count by Category", "Products"), use_container_width=True)
    b.plotly_chart(bar("avg_rating", "Average Rating by Category", "Avg Rating"), use_container_width=True)
    c, d = st.columns(2)
    c.plotly_chart(bar("avg_discount", "Average Discount by Category", "Avg Discount (%)"), use_container_width=True)
    d.plotly_chart(bar("avg_actual", "Average Actual Price by Category", "Avg Actual Price"), use_container_width=True)
    st.plotly_chart(bar("avg_discounted", "Average Discounted Price by Category", "Avg Discounted Price"),
                    use_container_width=True)

# ---------- Relationships ----------
with t3:
    sample = f.sample(min(len(f), 3000), random_state=1)  # keep scatter plots fast
    a, b = st.columns(2)
    a.plotly_chart(style(px.scatter(sample, x="actual_price", y="discounted_price", opacity=.6,
                   color="main_category", hover_name="product_name",
                   title="Actual vs Discounted Price")), use_container_width=True)
    b.plotly_chart(style(px.scatter(sample, x="rating", y="discounted_price", opacity=.6,
                   color="main_category", hover_name="product_name",
                   title="Rating vs Discounted Price")), use_container_width=True)
    c, d = st.columns(2)
    c.plotly_chart(style(px.scatter(sample, x="rating", y="rating_count", opacity=.6,
                   color="main_category", hover_name="product_name",
                   title="Rating vs Rating Count")), use_container_width=True)
    corr = f.select_dtypes(include=np.number).corr()
    d.plotly_chart(style(px.imshow(corr, text_auto=".2f", aspect="auto",
                   color_continuous_scale=[[0, "#146EB4"], [.5, NAVY], [1, ORANGE]],
                   zmin=-1, zmax=1, title="Correlation Heatmap")), use_container_width=True)

# ---------- Outliers ----------
with t4:
    col = st.selectbox("Column", ["actual_price", "discounted_price", "rating_count"])
    q1, q3 = f[col].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((f[col] < lo) | (f[col] > hi)).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Lower limit", f"{lo:,.0f}")
    m2.metric("Upper limit", f"{hi:,.0f}")
    m3.metric("Total outliers", f"{n_out:,}")
    st.plotly_chart(style(px.box(f, x=col, title=f"Outlier Detection – {col}",
                    color_discrete_sequence=[ORANGE]), 320), use_container_width=True)

# ---------- Top products ----------
with t5:
    n = st.slider("How many products?", 5, 25, 10)
    tables = {
        "🔥 Highest Discount": f.nlargest(n, "discount_percentage")[["product_name", "actual_price", "discount_percentage"]],
        "💰 Most Expensive": f.nlargest(n, "actual_price")[["product_name", "actual_price"]],
        "⭐ Most Popular (by rating count)": f.nlargest(n, "rating_count")[["product_name", "rating", "rating_count"]],
    }
    for title, t in tables.items():
        st.subheader(title)
        st.dataframe(t, use_container_width=True, hide_index=True)
