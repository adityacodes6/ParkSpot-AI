import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ParkSpot AI — Mumbai",
    page_icon="🅿️",
    layout="wide"
)

BASE = Path(__file__).parent
DATA_PATH = BASE / "mumbai_parking_clean.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    numeric = [
        "Latitude", "Longitude", "LMV_Capacity", "LCV_Capacity",
        "HMV_Capacity", "Total_Capacity", "Area_Lot"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Total_Capacity"] = df["Total_Capacity"].fillna(
        df[["LMV_Capacity", "LCV_Capacity", "HMV_Capacity"]].fillna(0).sum(axis=1)
    )
    return df

@st.cache_data
def add_clusters(df):
    work = df[["Total_Capacity", "Area_Lot"]].fillna(
        df[["Total_Capacity", "Area_Lot"]].median()
    )
    if len(df) >= 3:
        scaled = StandardScaler().fit_transform(work)
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)

        # Rename clusters according to their average capacity.
        means = pd.DataFrame({
            "label": labels,
            "capacity": df["Total_Capacity"].values
        }).groupby("label")["capacity"].mean().sort_values()

        names = {
            means.index[0]: "Small Capacity",
            means.index[1]: "Medium Capacity",
            means.index[2]: "Large Capacity"
        }
        df = df.copy()
        df["AI_Group"] = pd.Series(labels, index=df.index).map(names)
    else:
        df = df.copy()
        df["AI_Group"] = "Parking Group"
    return df

df = add_clusters(load_data())

# ---------- STYLE ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1d4ed8);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: .45rem 0 0; opacity: .9;}
.card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 1rem;
}
.small {font-size: .86rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🅿️ ParkSpot AI — Mumbai</h1>
<p>AI-assisted exploration of real BMC/MCGM parking locations using latitude, longitude and parking-capacity data.</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Parking Finder", "Mumbai Map", "Analytics", "AI Insights"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: BMC/MCGM Mumbai Parking KML provided for this project.")

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.subheader("Mumbai Parking Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parking Locations", f"{len(df):,}")
    c2.metric("Total Capacity", f"{int(df['Total_Capacity'].sum()):,}")
    c3.metric("Average Capacity", f"{df['Total_Capacity'].mean():.0f}")
    c4.metric("Largest Parking", f"{int(df['Total_Capacity'].max()):,}")

    st.markdown("### Dataset-based overview")
    st.info(
        "This dataset contains real Mumbai parking-location and capacity information. "
        "It does not contain historical occupied/vacant observations, so the app does "
        "not claim to predict live parking availability."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capacity by vehicle type")
        st.bar_chart(
            pd.DataFrame({
                "Capacity": [
                    df["LMV_Capacity"].fillna(0).sum(),
                    df["LCV_Capacity"].fillna(0).sum(),
                    df["HMV_Capacity"].fillna(0).sum()
                ]
            }, index=["LMV", "LCV", "HMV"])
        )

    with right:
        st.markdown("#### Largest parking locations")
        top = (
            df[["Parking_Name", "Total_Capacity"]]
            .sort_values("Total_Capacity", ascending=False)
            .head(10)
            .set_index("Parking_Name")
        )
        st.bar_chart(top)

# ---------- FINDER ----------
elif page == "Parking Finder":
    st.subheader("🔎 Mumbai Parking Finder")
    st.write("Select your requirements and find matching real parking locations.")

    c1, c2, c3 = st.columns(3)

    with c1:
        vehicle = st.selectbox(
            "Vehicle type",
            ["Any", "LMV", "LCV", "HMV"]
        )

    with c2:
        min_capacity = st.number_input(
            "Minimum total capacity",
            min_value=0,
            max_value=int(df["Total_Capacity"].max()),
            value=0,
            step=10
        )

    with c3:
        group = st.selectbox(
            "AI capacity group",
            ["Any"] + sorted(df["AI_Group"].dropna().unique().tolist())
        )

    if st.button("Find Parking", type="primary", use_container_width=True):
        result = df.copy()

        if vehicle != "Any":
            col = f"{vehicle}_Capacity"
            result = result[result[col].fillna(0) > 0]

        result = result[result["Total_Capacity"].fillna(0) >= min_capacity]

        if group != "Any":
            result = result[result["AI_Group"] == group]

        result = result.sort_values("Total_Capacity", ascending=False)

        st.session_state["finder_result"] = result

    result = st.session_state.get("finder_result")

    if result is not None:
        st.success(f"{len(result)} matching parking locations found.")

        if len(result) > 0:
            for _, row in result.head(10).iterrows():
                st.markdown(
                    f"**{row['Parking_Name']}** — "
                    f"Capacity: {int(row['Total_Capacity']):,} — "
                    f"Coordinates: {row['Latitude']:.6f}, {row['Longitude']:.6f}"
                )

            st.markdown("### Matching locations on Folium")
            center = [result["Latitude"].mean(), result["Longitude"].mean()]
            fmap = folium.Map(location=center, zoom_start=11, control_scale=True)

            for _, row in result.iterrows():
                popup = f"""
                <b>{row['Parking_Name']}</b><br>
                Code: {row['Parking_Code']}<br>
                Total Capacity: {int(row['Total_Capacity']) if pd.notna(row['Total_Capacity']) else 'N/A'}<br>
                LMV: {int(row['LMV_Capacity']) if pd.notna(row['LMV_Capacity']) else 0}<br>
                LCV: {int(row['LCV_Capacity']) if pd.notna(row['LCV_Capacity']) else 0}<br>
                HMV: {int(row['HMV_Capacity']) if pd.notna(row['HMV_Capacity']) else 0}
                """
                folium.Marker(
                    [row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(popup, max_width=300),
                    tooltip=row["Parking_Name"],
                    icon=folium.Icon(icon="car", prefix="fa")
                ).add_to(fmap)

            st_folium(fmap, width=None, height=560)
        else:
            st.warning("No parking location matches the selected filters.")

# ---------- MAP ----------
elif page == "Mumbai Map":
    st.subheader("🗺️ Real Mumbai Parking Locations")

    fmap = folium.Map(
        location=[19.0760, 72.8777],
        zoom_start=11,
        control_scale=True
    )

    for _, row in df.iterrows():
        popup = f"""
        <b>{row['Parking_Name']}</b><br>
        Code: {row['Parking_Code']}<br>
        Total Capacity: {int(row['Total_Capacity']) if pd.notna(row['Total_Capacity']) else 'N/A'}<br>
        Coordinates: {row['Latitude']:.6f}, {row['Longitude']:.6f}
        """
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            popup=folium.Popup(popup, max_width=300),
            tooltip=row["Parking_Name"],
            fill=True
        ).add_to(fmap)

    st_folium(fmap, width=None, height=650)

# ---------- ANALYTICS ----------
elif page == "Analytics":
    st.subheader("📊 Parking Analytics")

    a, b = st.columns(2)

    with a:
        st.markdown("#### Total capacity distribution")
        st.bar_chart(
            df["Total_Capacity"].value_counts().sort_index().head(20)
        )

    with b:
        st.markdown("#### Parking capacity groups")
        st.bar_chart(df["AI_Group"].value_counts())

    st.markdown("#### Capacity vs parking-lot area")
    chart_df = df[["Area_Lot", "Total_Capacity"]].dropna()
    st.scatter_chart(chart_df, x="Area_Lot", y="Total_Capacity")

# ---------- AI INSIGHTS ----------
else:
    st.subheader("🤖 AI Insights")

    st.info(
        "The uploaded Mumbai dataset does not contain a target such as occupied/vacant "
        "or historical availability. Therefore, a supervised occupancy-prediction model "
        "would be misleading. Instead, ParkSpot AI uses K-Means clustering to group "
        "real parking locations by capacity and parking-lot area."
    )

    summary = (
        df.groupby("AI_Group")
        .agg(
            Parking_Locations=("Parking_Name", "count"),
            Average_Capacity=("Total_Capacity", "mean"),
            Average_Area=("Area_Lot", "mean")
        )
        .reset_index()
    )

    summary["Average_Capacity"] = summary["Average_Capacity"].round(0)
    summary["Average_Area"] = summary["Average_Area"].round(0)

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### How the AI component works")
    st.markdown("""
    1. The real Mumbai parking records are loaded.
    2. `Total_Capacity` and `Area_Lot` are standardized.
    3. K-Means groups parking locations into three capacity/area groups.
    4. The groups are used by the Parking Finder to filter and visualize suitable locations.
    """)
