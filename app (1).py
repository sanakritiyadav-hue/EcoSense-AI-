
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# PAGE SETTINGS
st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌱",
    layout="wide"
)

# ENVIRONMENTAL DATA
data = {
    "Date": pd.date_range("2026-11-01", periods=10),
    "PM2.5": [45, 48, 52, 110, 55, 50, 47, 53, 49, 51],
    "PM10": [80, 85, 90, 160, 95, 88, 82, 92, 86, 89],
    "Water_Usage": [1200, 1250, 1300, 1800, 1280,
                    1250, 1220, 1350, 1270, 1290],
    "Energy_Usage": [450, 460, 470, 620, 465,
                     455, 448, 475, 460, 468]
}

df = pd.DataFrame(data)

# MACHINE LEARNING ANALYSIS
features = df[
    ["PM2.5", "PM10", "Water_Usage", "Energy_Usage"]
]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

model = IsolationForest(
    contamination=0.1,
    random_state=42
)

df["AI_Result"] = model.fit_predict(scaled_data)

df["Status"] = df["AI_Result"].map({
    1: "Normal",
    -1: "Unusual"
})

# SIDEBAR
st.sidebar.title("🌱 EcoSense AI")
st.sidebar.caption("Smart Environmental Monitoring")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Air Quality",
        "Water Monitoring",
        "Energy Monitoring",
        "AI Insights",
        "Reports"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Environmental Intelligence Platform")

# HOME DASHBOARD
if page == "Home Dashboard":

    st.title("🌱 EcoSense AI Dashboard")
    st.write("Welcome to your Smart Environmental Monitoring System.")

    st.info("Demonstration dashboard using sample environmental data.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average PM2.5", f"{df['PM2.5'].mean():.1f}")
    col2.metric("Average PM10", f"{df['PM10'].mean():.1f}")
    col3.metric("Water Usage", f"{df['Water_Usage'].mean():.0f} L")
    col4.metric("Energy Usage", f"{df['Energy_Usage'].mean():.0f} kWh")

    st.subheader("Environmental Trends")

    fig = px.line(
        df,
        x="Date",
        y=["PM2.5", "PM10"],
        title="Air Quality Trends",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Environmental Data")
    st.dataframe(df, use_container_width=True)

# AIR QUALITY
elif page == "Air Quality":

    st.title("🌫️ Air Quality Monitoring")

    col1, col2 = st.columns(2)

    col1.metric("Average PM2.5", f"{df['PM2.5'].mean():.1f}")
    col2.metric("Average PM10", f"{df['PM10'].mean():.1f}")

    fig = px.line(
        df,
        x="Date",
        y=["PM2.5", "PM10"],
        markers=True,
        title="Air Pollution Monitoring"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("PM2.5 and PM10 measurements help monitor particulate matter in air.")

# WATER MONITORING
elif page == "Water Monitoring":

    st.title("💧 Water Consumption Monitoring")

    st.metric(
        "Average Water Usage",
        f"{df['Water_Usage'].mean():.0f} Litres"
    )

    fig = px.bar(
        df,
        x="Date",
        y="Water_Usage",
        title="Daily Water Consumption"
    )

    st.plotly_chart(fig, use_container_width=True)


    st.write("Monitor water consumption and identify unusual usage.")

    
    # Smart Water Usage Analysis
    water_avg = df["Water_Usage"].mean()
    water_std = df["Water_Usage"].std()
    unusual_water = df[df["Water_Usage"] > water_avg + water_std]

    highest_row = df.loc[df["Water_Usage"].idxmax()]
    highest_usage = highest_row["Water_Usage"]
    increase = ((highest_usage - water_avg) / water_avg) * 100

    st.subheader("Smart Water Usage Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Daily Usage", f"{water_avg:.0f} L")
    col2.metric("Highest Usage", f"{highest_usage:.0f} L")
    col3.metric("Increase Above Average", f"{increase:.1f}%")

    st.write("**Date of Highest Consumption:**", highest_row["Date"])

    if not unusual_water.empty:
        st.warning("Unusual water consumption detected!")
        st.write("**Unusual Usage Records:**")
        st.dataframe(unusual_water[["Date", "Water_Usage"]])

        st.write("**Possible Reasons:**")
        st.write("- Water leakage")
        st.write("- Excessive water usage")
        st.write("- Unnecessary water wastage")

        st.write("**Practical Recommendations:**")
        st.write("- Check pipes and taps for leakage.")
        st.write("- Monitor daily water consumption.")
        st.write("- Reduce unnecessary water usage.")
    else:
        st.success("No unusual water consumption detected.")

    st.info("This analysis uses sample demonstration data, not live sensor readings.")



# ENERGY MONITORING
elif page == "Energy Monitoring":

    st.title("⚡ Energy Consumption Monitoring")

    st.metric(
        "Average Energy Usage",
        f"{df['Energy_Usage'].mean():.1f} kWh"
    )

    fig = px.line(
        df,
        x="Date",
        y="Energy_Usage",
        markers=True,
        title="Daily Energy Consumption"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("Monitor electricity usage and identify unusual increases.")

# AI INSIGHTS
elif page == "AI Insights":

    st.title("🤖 AI Environmental Insights")

    unusual = df[df["Status"] == "Unusual"]

    st.subheader("Detected Environmental Anomalies")

    st.dataframe(unusual, use_container_width=True)

    st.warning(
        f"The machine learning model detected {len(unusual)} "
        "unusual reading(s) in the demonstration dataset."
    )

    st.subheader("Gemini AI Environmental Analysis")

    import os
    from google import genai

    try:
        from google.colab import userdata
        api_key = userdata.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        st.error("Gemini API key is not available.")
    else:
        try:
            gemini_client = genai.Client(api_key=api_key)

            prompt = f"""
            Analyse these unusual environmental readings:

            {unusual.to_string(index=False)}

            Write a report using these five headings:
            1. Unusual Environmental Changes
            2. Possible Environmental Risks
            3. Possible Reasons
            4. Practical Recommendations
            5. Overall Environmental Summary

            Use simple English and practical suggestions.
            """

            response = None
            last_error = None

            for model_name in ["gemini-3.5-flash-lite", "gemini-3.6-flash"]:
                try:
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    break
                except Exception as error:
                    last_error = error

            if response is not None:
                st.markdown(response.text)
            else:
                st.error(f"Gemini generation failed: {last_error}")

        except Exception as error:
            st.error(f"AI connection error: {error}")

    st.success("EcoSense AI helps users understand environmental changes.")

# REPORTS
elif page == "Reports":

    st.title("📊 Environmental Reports")

    st.subheader("Environmental Data Report")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Environmental Report (CSV)",
        data=csv,
        file_name="EcoSense_Environmental_Report.csv",
        mime="text/csv"
    )

    st.success("Your environmental report is ready to download.")
