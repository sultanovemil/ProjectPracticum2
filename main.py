import streamlit as st
import requests
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import pandas as pd
from datetime import datetime
import holidays
import numpy as np
import pickle
import xgboost as xgb

# Setting up the page configuration for Streamlit App
st.set_page_config(
    page_title="Taxi",
    # layout="wide",
    initial_sidebar_state="expanded"
)


# Load the XGBoost model
# @st.cache_data()
def get_model():
    model = pickle.load(open("models/model_xgb.pkl", "rb"))
    return model


# Function to make prediction using the model and input data
def make_prediction(data):
    model = get_model()
    best_features = [
        'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
        'pickup_hour', 'pickup_holiday', 'total_distance', 'total_travel_time',
        'number_of_steps', 'haversine_distance', 'temperature',
        'pickup_day_of_week_1', 'pickup_day_of_week_2', 'pickup_day_of_week_3',
        'pickup_day_of_week_4', 'pickup_day_of_week_5', 'pickup_day_of_week_6',
        'geo_cluster_1', 'geo_cluster_3', 'geo_cluster_5', 'geo_cluster_7',
        'geo_cluster_9']
    data_matrix = xgb.DMatrix(data, feature_names=best_features)
    return model.predict(data_matrix)


# Get coordinates from address
def get_coordinates(address):
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(address)
    return (location.longitude, location.latitude)


def show_map(lon_from, lat_from, lon_to, lat_to):
    # Creating a map
    fig = go.Figure(go.Scattermapbox(
        mode="markers",
        marker={'size': 15, 'color': 'red'}
    ))

    # Adding markers
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[lon_from, lon_to],
        lat=[lat_from, lat_to],
        marker=go.scattermapbox.Marker(
            size=25,
            color='red'
        )
    ))

    # Adding a line
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=[lon_from, lon_to],
        lat=[lat_from, lat_to],
        line=dict(width=2, color='green')
    ))

    # Configuring the display of a map
    fig.update_layout(
        mapbox={
            'style': "open-street-map",
            'center': {
                'lon': (lon_from + lon_to) / 2, 'lat': (lat_from + lat_to) / 2
            },
            'zoom': 9},
        showlegend=False,
        height=600,
        width=1200
    )

    # Display the map
    return fig


# Get total distance
def get_total_distance(
        start_longitude, start_latitude, end_longitude, end_latitude):
    # Construct the URL for sending a request to the public OSRM server
    url = f"http://router.project-osrm.org/route/v1/driving/{start_longitude},{start_latitude};{end_longitude},{end_latitude}?overview=false"

    # Send a GET request to the OSRM server
    response = requests.get(url)

    # Process the response from the server
    if response.status_code == 200:
        data = response.json()
        # Total distance in meters
        total_distance = data["routes"][0]["distance"]
        # Total travel time in seconds
        total_travel_time = data["routes"][0]["duration"]
        # Number of steps in the
        number_of_steps = len(data["routes"][0]["legs"][0]["steps"])
        return total_distance,  total_travel_time, number_of_steps


# Get Harversine distance
def get_haversine_distance(lat1, lng1, lat2, lng2):
    # Convert angles to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # Earth's radius in kilometers
    EARTH_RADIUS = 6371
    # Calculate the shortest distance h using the Haversine formula
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# User input features
def user_input_features(lon_from, lat_from, lon_to, lat_to, passenger_count):
    current_time = datetime.now()
    pickup_hour = current_time.hour
    today = datetime.today()
    pickup_holiday = 1 if today in holidays.USA() else 0
    total_distance,  total_travel_time, number_of_steps = get_total_distance(
        lon_from, lat_from, lon_to, lat_to)
    haversine_distance = get_haversine_distance(
        lat_from, lon_from, lat_to, lon_to)
    weekday_number = current_time.weekday()

    data = {'vendor_id': 1,
            'passenger_count': passenger_count,
            'pickup_longitude': lon_from,
            'pickup_latitude': lat_from,
            'dropoff_longitude': lon_to,
            'dropoff_latitude': lat_to,
            'store_and_fwd_flag': 0.0,
            'pickup_hour': pickup_hour,
            'pickup_holiday': pickup_holiday,
            'total_distance': total_distance,
            'total_travel_time': total_travel_time,
            'number_of_steps': number_of_steps,
            'haversine_distance': haversine_distance,
            'temperature': 15,
            'pickup_day_of_week_1': 1 if weekday_number == 1 else 0,
            'pickup_day_of_week_2': 1 if weekday_number == 2 else 0,
            'pickup_day_of_week_3': 1 if weekday_number == 3 else 0,
            'pickup_day_of_week_4': 1 if weekday_number == 4 else 0,
            'pickup_day_of_week_5': 1 if weekday_number == 5 else 0,
            'pickup_day_of_week_6': 1 if weekday_number == 6 else 0,
            'geo_cluster_1': 1,
            'geo_cluster_3': 0,
            'geo_cluster_5': 0,
            'geo_cluster_7': 0,
            'geo_cluster_9': 0
            }
    features = pd.DataFrame(data, index=[0])
    return features


# Scale the input data using a pre-trained MinMaxScaler
def min_max_scaler(data):
    scaler = pickle.load(open("models/min_max_scaler.pkl", "rb"))
    data_scaled = scaler.transform(data)
    return data_scaled


# Main function
def main():
    if 'btn_predict' not in st.session_state:
        st.session_state['btn_predict'] = False

    # Sidebar
    st.sidebar.markdown(''' # New York City Taxi Trip Duration''')
    st.sidebar.image("img/taxi_img.png")
    address_from = st.sidebar.text_input(
        "Откуда:", value="New York, 11 Wall Street")
    address_to = st.sidebar.text_input(
        "Куда:", value="New York, 740 Park Avenue")
    passenger_count = st.sidebar.slider("Количество пассажиров", 1, 4, 1)

    st.session_state['btn_predict'] = st.sidebar.button('Start')

    if st.session_state['btn_predict']:
        lon_from, lat_from = get_coordinates(address_from)
        lon_to, lat_to = get_coordinates(address_to)
        st.plotly_chart(show_map(lon_from, lat_from, lon_to, lat_to))
        user_data = user_input_features(
            lon_from, lat_from, lon_to, lat_to, passenger_count)
        # st.write(user_data)
        data_scaled = min_max_scaler(user_data)
        trip_duration = np.exp(make_prediction(data_scaled)) - 1
        trip_duration = round(float(trip_duration) / 60)
        st.markdown(f"""
                        <div style='background-color: lightgreen; padding: 10px;'>
                            <h2 style='color: black; text-align: center;'>Длительность поездки составит: {trip_duration} мин.</h2>
                        </div>
                    """, unsafe_allow_html=True)


# Running the main function
if __name__ == "__main__":
    main()
