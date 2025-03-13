import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from xgboost import XGBClassifier
import plotly.express as px
import base64

# Ensure set_page_config is the first command
st.set_page_config(layout="wide")

# Load Dataset (Replace with actual dataset path)
data_path = "dummy_npi_data.xlsx"
df = pd.read_excel(data_path)

# Preprocessing
df['Login Time'] = pd.to_datetime(df['Login Time'])
df['Logout Time'] = pd.to_datetime(df['Logout Time'])
df['Login Hour'] = df['Login Time'].dt.hour
df['Logout Hour'] = df['Logout Time'].dt.hour

# Features and Target
X = df[['Login Hour', 'Usage Time (mins)', 'Count of Survey Attempts']]
y = (df['Count of Survey Attempts'] > df['Count of Survey Attempts'].median()).astype(int)

# Train Model
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X, y)

# Save Model
joblib.dump(model, 'doctor_survey_model.pkl')

# Set full background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        [data-testid="stSidebar"] {{
            background-color: #0D1B2A !important;
        }}
        .stRadio > div {{
            color: white !important;
            font-weight: bold !important;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("home_backgroundimg.jpg")  # Ensure the correct image file path

# Sidebar with Dark Blue Background
with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #0D1B2A !important;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown("<h1 style='color:white;'>INSIGHTS</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .sidebar-button {
        background-color: #0D1B2A;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        width: 100%;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
if "page" not in st.session_state:
    st.session_state.page = "🌐 Home"  # Default page

# Function to switch pages (no need for st.rerun)
def switch_page(page_name):
    st.session_state.page = page_name

st.sidebar.button("🌐 Home", on_click=lambda: switch_page("🌐 Home"))
st.sidebar.button("📶 Data Insights", on_click=lambda: switch_page("📶 Data Insights"))
st.sidebar.button("📑🔎 Result Insights", on_click=lambda: switch_page("📑🔎 Result Insights"))



page = st.session_state.get("page", "🌐 Home")

# Home Page
best_doctors = None 
input_hour = None # Initialize best_doctors to avoid errors
# Ensure session state for best_doctors
# Ensure session state for best_doctors
if "best_doctors" not in st.session_state:
    st.session_state.best_doctors = None

if "input_hour" not in st.session_state:
    st.session_state.input_hour = None

if page == "🌐 Home":
    st.title("Doctor Survey Prediction📋🩺👩🏻‍⚕️💉")
    st.write("Enter a time to get the list of doctors most likely to attend the survey.")
    
    # User Input
    time_input = st.time_input("Select Time ⏱", value=None)
    if st.button("Predict 🛎️", key="predict_button") and time_input is not None:
        input_hour = time_input.hour
    
        # Load Model
        model = joblib.load('doctor_survey_model.pkl')
        
        # Filter doctors who are active at the given hour
        active_doctors = df[(df['Login Hour'] <= input_hour) & (df['Logout Hour'] >= input_hour)]
        
        if not active_doctors.empty:
            active_doctors['Prediction'] = model.predict(active_doctors[['Login Hour', 'Usage Time (mins)', 'Count of Survey Attempts']])
            st.session_state.best_doctors = active_doctors[active_doctors['Prediction'] == 1][['NPI', 'Speciality', 'Region','State']]
        else:
            st.session_state.best_doctors = pd.DataFrame(columns=['NPI', 'Speciality', 'Region','State'])
        
        # Display & Export
        if not st.session_state.best_doctors.empty:
            st.dataframe(st.session_state.best_doctors)
            st.download_button("Download CSV", st.session_state.best_doctors.to_csv(index=False), "doctors_list.csv", "text/csv")

# Data Insights Page
elif page == "📶 Data Insights":
    st.title("Data Insights💡")
    
    # Doctor Activity Over Time
    st.subheader("Doctor Activity Over Time ⏳")
    hourly_activity = df.groupby("Login Hour")["NPI"].count().reset_index()
    hourly_activity.columns = ["Hour", "Active Doctors"]
    fig = px.line(hourly_activity, x="Hour", y="Active Doctors", markers=True, title="Doctors Logged In by Hour")
    st.plotly_chart(fig)
    st.subheader("Specialty Distribution")
    fig_pie = px.pie(df, names="Speciality", title="Doctor Specialties Distribution")
    st.plotly_chart(fig_pie)

    # Statewise Distribution
    st.subheader("Statewise Distribution")
    df_s = df.groupby("State")["NPI"].count().reset_index()
    df_s.columns = ["State", "Count"]  # Ensure correct column renaming
    fig_bar_s = px.bar(df_s, x="State", y="Count", color="State", title="Doctor State Distribution", color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_bar_s)

    # Regionwise Distribution
    st.subheader("Regionwise Distribution")
    df_r = df.groupby("Region")["NPI"].count().reset_index()
    df_r.columns = ["Region", "Count"]  # Ensure correct column renaming
    fig_bar_r = px.bar(df_r, x="Region", y="Count", color="Region", title="Doctor Region Distribution",color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_bar_r)

    st.subheader("Doctor Activity Over Different Times of the Day")

    # Categorizing login hours into time of day
    def categorize_time(hour):
        if 5 <= hour <= 11:
            return "Morning"
        elif 12 <= hour <= 16:
            return "Afternoon"
        elif 17 <= hour <= 19:
            return "Evening"
        else:
            return "Night"

    # Ensure 'Login Hour' is in integer format
    df["Time of Day"] = df["Login Hour"].astype(int).apply(categorize_time)

    # Count occurrences of each time category
    df_time = df["Time of Day"].value_counts().reset_index()
    df_time.columns = ["Time of Day", "Count"]

    # Sorting the time categories for better visualization
    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    df_time["Time of Day"] = pd.Categorical(df_time["Time of Day"], categories=time_order, ordered=True)
    df_time = df_time.sort_values("Time of Day")

    # Plot line chart
    fig_line_time = px.line(df_time, x="Time of Day", y="Count", markers=True,
                            title="Doctor Activity Over Different Times of the Day",
                            line_shape="linear")
    st.plotly_chart(fig_line_time)

# Result Insights Page
elif page == "📑🔎 Result Insights":
    st.title("Result Insights🎯 ")

    if st.session_state.input_hour is not None and not st.session_state.input_hour.empty:
        st.subheader(input_hour)
    # Specialty Distribution
    #st.subheader("Specialty Distribution")
    
    if st.session_state.best_doctors is not None and not st.session_state.best_doctors.empty:
        st.subheader("Specialty Distribution")
        fig_pie = px.pie(st.session_state.best_doctors, names="Speciality", title="Doctor Specialties Distribution")
        st.plotly_chart(fig_pie)

    # Statewise Distribution
        st.subheader("Statewise Distribution📍")
        df_s = st.session_state.best_doctors["State"].value_counts().reset_index()
        df_s.columns = ["State", "Count"]  # Ensure correct column renaming
        fig_bar_s = px.bar(df_s, x="State", y="Count", title="Doctor State Distribution")
        st.plotly_chart(fig_bar_s)

    # Regionwise Distribution
        st.subheader("Regionwise Distribution")
        df_r = st.session_state.best_doctors["Region"].value_counts().reset_index()
        df_r.columns = ["Region", "Count"]  # Ensure correct column renaming
        fig_bar_r = px.bar(df_r, x="Region", y="Count", title="Doctor Region Distribution")
        st.plotly_chart(fig_bar_r)


    else:
        st.warning("No doctor data available. Please predict first in the Home page.")


