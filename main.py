# --- IMPORTS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import streamlit as st

# --- DATA GENERATION ---
fake = Faker()
num_rows = 5000
np.random.seed(42)

data = {
    'ID': [f"A{str(i).zfill(5)}" for i in range(1, num_rows + 1)],
    'Source': [random.choice(['US', 'UK', 'CA', 'AU']) for _ in range(num_rows)],
    'Severity': [random.choice([1, 2, 3]) for _ in range(num_rows)],
    'Start_Time': [str(fake.date_this_decade()) + ' ' + fake.time() for _ in range(num_rows)],
    'End_Time': [str(fake.date_this_decade()) + ' ' + fake.time() for _ in range(num_rows)],
    'Start_Lat': np.random.uniform(-90, 90, num_rows),
    'Start_Lng': np.random.uniform(-180, 180, num_rows),
    'End_Lat': np.random.uniform(-90, 90, num_rows),
    'End_Lng': np.random.uniform(-180, 180, num_rows),
    'Distance(mi)': np.random.uniform(0, 50, num_rows),
    'Description': [fake.text(max_nb_chars=50) for _ in range(num_rows)],
    'Street': [fake.street_name() for _ in range(num_rows)],
    'City': [fake.city() for _ in range(num_rows)],
    'State': [fake.state() for _ in range(num_rows)],
    'Zipcode': [fake.zipcode() for _ in range(num_rows)],
    'Country': [fake.country() for _ in range(num_rows)],
    'Timezone': [random.choice(['PST', 'EST', 'CST', 'MST']) for _ in range(num_rows)],
    'Airport_Code': [random.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW']) for _ in range(num_rows)],
    'Weather_Timestamp': [str(fake.date_this_decade()) + ' ' + fake.time() for _ in range(num_rows)],
    'Temperature(F)': np.random.uniform(-30, 100, num_rows),
    'Wind_Chill(F)': np.random.uniform(-50, 50, num_rows),
    'Humidity(%)': np.random.uniform(0, 100, num_rows),
    'Weather_Condition': [random.choice(['Clear', 'Rain', 'Fog', 'Snow', 'Cloudy']) for _ in range(num_rows)],
}

df = pd.DataFrame(data)
df.to_csv("simulated_traffic_data_50001.csv", index=False)

# --- DATA PROCESSING + ML ---
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])
df['Accident_Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

features = ['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Accident_Duration']
X = df[features]
y = df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- STREAMLIT DASHBOARD FUNCTION ---
def run_dashboard():
    df = pd.read_csv("simulated_traffic_data_50001.csv")

    st.title("🚦 Traffic Accident Analysis Dashboard")
    st.sidebar.header("🔎 Filters")

    severity_filter = st.sidebar.multiselect("Select Accident Severity:", options=sorted(df['Severity'].unique()))
    weather_filter = st.sidebar.multiselect("Select Weather Condition:", options=sorted(df['Weather_Condition'].unique()))

    # Safe condition handling
    condition = pd.Series([True] * len(df))
    if severity_filter:
        condition &= df['Severity'].isin(severity_filter)
    if weather_filter:
        condition &= df['Weather_Condition'].isin(weather_filter)

    filtered_data = df[condition]

    st.subheader("📋 Filtered Data")
    st.write(filtered_data)

    st.subheader("📊 Accident Severity Distribution")
    st.bar_chart(filtered_data['Severity'].value_counts())

    st.subheader("🌦️ Accidents by Weather Condition")
    st.bar_chart(filtered_data['Weather_Condition'].value_counts())

# --- UNCOMMENT BELOW LINE TO RUN STREAMLIT DASHBOARD ---
run_dashboard()
