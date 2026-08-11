import streamlit as st
import pandas as pd
import pickle
import random
import time

# Load AI model
with open("smart_traffic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Traffic Signal AI")
st.write("Real-time traffic simulation and AI signal optimization")

st.divider()

# Simulation settings
st.header("⚙️ Simulation")

duration = st.slider(
    "Simulation Duration (seconds)",
    5,
    60,
    20
)

start = st.button(
    "▶️ Start Traffic Simulation",
    use_container_width=True
)

if start:

    roads = ["Road A", "Road B", "Road C", "Road D"]

    progress = st.progress(0)

    status = st.empty()

    table_area = st.empty()

    chart_area = st.empty()

    for second in range(duration):

        results = []

        for road in roads:

            # Generate simulated traffic
            vehicles = random.randint(10, 120)
            waiting_time = random.randint(5, 60)

            # Advanced features
            traffic_density = vehicles / (waiting_time + 1)

            traffic_pressure = (
                vehicles * 0.7 +
                waiting_time * 0.3
            )

            input_data = pd.DataFrame([{
                "vehicles": vehicles,
                "waiting_time": waiting_time,
                "traffic_density": traffic_density,
                "traffic_pressure": traffic_pressure
            }])

            # AI prediction
            prediction = model.predict(input_data)[0]

            signal_time = max(
                10,
                min(60, prediction)
            )

            # Traffic level
            if vehicles >= 80:
                traffic_level = "HIGH"

            elif vehicles >= 40:
                traffic_level = "MEDIUM"

            else:
                traffic_level = "LOW"

            results.append({
                "Road": road,
                "Vehicles": vehicles,
                "Waiting Time": waiting_time,
                "Traffic": traffic_level,
                "Green Time": round(signal_time)
            })

        df = pd.DataFrame(results)

        # Highest traffic road
        priority = df.loc[
            df["Vehicles"].idxmax()
        ]

        status.info(
            f"🚨 Priority Road: {priority['Road']} | "
            f"Vehicles: {priority['Vehicles']} | "
            f"Green Time: {priority['Green Time']} sec"
        )

        table_area.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        chart_area.bar_chart(
            df.set_index("Road")["Vehicles"]
        )

        progress.progress(
            (second + 1) / duration
        )

        time.sleep(1)

    st.success("✅ Traffic simulation completed!")