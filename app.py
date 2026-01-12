import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import geopandas as gpd
import json
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
from sklearn.cluster import DBSCAN

# Set page config with responsive settings
st.set_page_config(
    page_title="Aadhaar Analytics Dashboard",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="auto"  # Auto collapse on mobile
)

# Custom CSS for responsive design
st.markdown("""
<style>
    /* Base styles for all devices */
    .main-header {color: #1E88E5; margin-bottom: 15px; font-size: 28px;}
    .section-header {color: #1E88E5; margin: 15px 0 10px 0; font-size: 20px;}
    .metric-card {background-color: #f8f9fa; border-radius: 10px; padding: 12px; margin: 8px 0;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .stSelectbox, .stSlider, .stDateInput {margin-bottom: 12px;}
    
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main-header {font-size: 24px; margin-bottom: 12px;}
        .section-header {font-size: 18px; margin: 12px 0 8px 0;}
        .metric-card {padding: 10px; margin: 6px 0;}
        .stSelectbox, .stSlider, .stDateInput {margin-bottom: 8px;}
        
        /* Adjust layout for mobile */
        .block-container {padding: 1rem 1rem 1rem 1rem;}
        .st-emotion-cache-1y4p3pa {padding: 1rem 1rem 1rem 1rem;}
        
        /* Make tables scroll horizontally on mobile */
        .stDataFrame {width: 100%; overflow-x: auto;}
    }
    
    /* Desktop styles */
    @media (min-width: 769px) {
        .main-header {font-size: 36px;}
        .section-header {font-size: 24px;}
        .metric-card {padding: 15px; margin: 10px 0;}
    }
    
    /* Common responsive elements */
    .stPlotlyChart {
        width: 100% !important;
        max-width: 100%;
        height: auto !important;
    }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        min-width: 200px;
        max-width: 300px;
    }
    
    /* Make sure images are responsive */
    img {
        max-width: 100%;
        height: auto;
    }
    
    /* Mobile menu button */
    @media (max-width: 768px) {
        .mobile-menu-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            background: #1E88E5;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Hide sidebar by default on mobile */
        [data-testid="stSidebar"] {
            transform: translateX(-100%);
            transition: transform 0.3s ease-in-out;
        }
        
        [data-testid="stSidebar"].sidebar-visible {
            transform: translateX(0);
        }
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate comprehensive sample data with realistic patterns"""
    np.random.seed(42)
    
    # Extended date range: 3 years of data (2024-2026)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Indian states with coordinates for mapping
    states_info = {
        'Maharashtra': {'lat': 19.7515, 'lon': 75.7139, 'population': 123144223},
        'Uttar Pradesh': {'lat': 26.8467, 'lon': 80.9462, 'population': 199812341},
        'Karnataka': {'lat': 15.3173, 'lon': 75.7139, 'population': 61095297},
        'Tamil Nadu': {'lat': 11.1271, 'lon': 78.6569, 'population': 72147030},
        'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'population': 16787941},
        'West Bengal': {'lat': 22.9868, 'lon': 87.8550, 'population': 91276115},
        'Gujarat': {'lat': 22.2587, 'lon': 71.1924, 'population': 60439692},
        'Rajasthan': {'lat': 27.0238, 'lon': 74.2179, 'population': 68548437},
        'Bihar': {'lat': 25.0961, 'lon': 85.3131, 'population': 104099452},
        'Andhra Pradesh': {'lat': 15.9129, 'lon': 79.7400, 'population': 49577103},
        'Madhya Pradesh': {'lat': 22.9734, 'lon': 78.6569, 'population': 72626809},
        'Kerala': {'lat': 10.8505, 'lon': 76.2711, 'population': 33406061},
        'Punjab': {'lat': 31.1471, 'lon': 75.3412, 'population': 27743338},
        'Haryana': {'lat': 29.0588, 'lon': 76.0856, 'population': 25351462},
        'Odisha': {'lat': 20.9517, 'lon': 85.0985, 'population': 41974218}
    }
    
    states = list(states_info.keys())
    n_records = 10000  # Increased sample size for better analysis
    
    # Generate dates with seasonality
    dates = []
    for _ in range(n_records):
        # Weight dates to create realistic patterns
        year = np.random.choice([2024, 2025, 2026], p=[0.3, 0.4, 0.3])
        month = np.random.choice(range(1, 13), p=[0.12, 0.08, 0.1, 0.08, 0.07, 0.06, 0.05, 0.06, 0.09, 0.08, 0.09, 0.12])
        day = np.random.randint(1, 29 if month == 2 else 31 if month in [4,6,9,11] else 32)
        hour = np.random.normal(14, 3, 1)[0] % 24  # More activity during business hours
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        try:
            dates.append(datetime(year, month, day, int(hour), minute, second))
        except ValueError:
            # Handle invalid dates (like Feb 30)
            dates.append(datetime(year, month, 28, int(hour), minute, second))
    
    # Sort dates to make time series analysis meaningful
    dates.sort()
    
    # Generate realistic age distribution
    def get_age_distribution(n):
        # Bimodal distribution for age
        ages1 = np.random.normal(25, 5, n//2).astype(int)
        ages2 = np.random.normal(55, 10, n - n//2).astype(int)
        ages = np.concatenate([ages1, ages2])
        return np.clip(ages, 1, 100)
    
    # Generate data
    data = {
        'Aadhaar_Number': [f"{np.random.randint(1000, 10000):04d} {np.random.randint(1000, 10000):04d} {np.random.randint(1000, 10000):04d}" for _ in range(n_records)],
        'Enrollment_Date': dates,
        'Enrollment_Type': np.random.choice(['New Enrollment', 'Update'], n_records, p=[0.7, 0.3]),
        'Update_Type': np.random.choice(['Address', 'Mobile', 'Biometric', 'Name', 'DOB', 'Photo'], n_records, p=[0.25, 0.3, 0.15, 0.1, 0.1, 0.1]),
        'Update_Reason': np.random.choice(['Change of Address', 'Lost Card', 'Damaged Card', 'Data Correction', 'First Time', 'Biometric Update'], n_records, p=[0.3, 0.2, 0.15, 0.2, 0.1, 0.05]),
        'State': np.random.choice(states, n_records, p=[s['population']/sum(info['population'] for info in states_info.values()) for s in states_info.values()]),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.51, 0.48, 0.01]),
        'Age': get_age_distribution(n_records),
        'Timestamp': [date + timedelta(seconds=np.random.randint(0, 86400)) for date in dates],
        'Status': np.random.choice(['Completed', 'In Progress', 'Pending Verification', 'Rejected'], n_records, p=[0.85, 0.08, 0.05, 0.02]),
        'Processing_Time_Minutes': np.random.lognormal(3, 0.5, n_records).clip(1, 240)  # Processing time in minutes
    }
    
    df = pd.DataFrame(data)
    
    # Set Update_Type to None for non-update records
    df['Update_Type'] = df.apply(lambda x: x['Update_Type'] if x['Enrollment_Type'] == 'Update' else None, axis=1)
    
    # Add coordinates for mapping
    df['Latitude'] = df['State'].apply(lambda x: states_info[x]['lat'] + np.random.normal(0, 0.5))
    df['Longitude'] = df['State'].apply(lambda x: states_info[x]['lon'] + np.random.normal(0, 0.5))
    
    # Add time-based features
    df['Year'] = df['Enrollment_Date'].dt.year
    df['Month'] = df['Enrollment_Date'].dt.month
    df['Day_Of_Week'] = df['Enrollment_Date'].dt.dayofweek
    df['Hour_Of_Day'] = df['Timestamp'].dt.hour
    df['Day_Name'] = df['Enrollment_Date'].dt.day_name()
    
    # Add age groups
    age_bins = [0, 18, 25, 35, 50, 65, 100]
    age_labels = ['0-18', '19-25', '26-35', '36-50', '51-65', '65+']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # Add some realistic patterns
    # More enrollments on weekdays
    df['Weekend'] = df['Day_Of_Week'].isin([5, 6])
    
    # More updates in certain months
    df['Update_Season'] = df['Month'].apply(
        lambda x: 'High' if x in [1, 7, 12] else 'Medium' if x in [3, 6, 9] else 'Low'
    )
    
    return df

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    try:
        # Try to load the data from CSV
        df = pd.read_csv("aadhaar_data.csv")
        
        # Check if required columns exist
        required_columns = ['Enrollment_Date', 'Enrollment_Type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Missing required columns in data: {', '.join(missing_columns)}. Using sample data instead.")
            return generate_sample_data()
            
        # Convert date and extract features
        df['Enrollment_Date'] = pd.to_datetime(df['Enrollment_Date'])
        
        # Create additional date-based columns
        df['Year'] = df['Enrollment_Date'].dt.year
        df['Month'] = df['Enrollment_Date'].dt.month
        df['Day_Of_Week'] = df['Enrollment_Date'].dt.dayofweek
        
        # Handle Timestamp if it exists, otherwise use Enrollment_Date
        if 'Timestamp' in df.columns and not df['Timestamp'].isna().all():
            df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        else:
            df['Hour'] = df['Enrollment_Date'].dt.hour
        
        # Ensure numeric columns are properly typed
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # Set default values for optional columns if they don't exist
        if 'State' not in df.columns:
            df['State'] = 'Unknown'
        if 'Gender' not in df.columns:
            df['Gender'] = 'Unknown'
        if 'Age' not in df.columns:
            df['Age'] = np.random.normal(35, 15, len(df)).clip(1, 100).astype(int)
        if 'Enrollment_Type' not in df.columns:
            df['Enrollment_Type'] = np.random.choice(['New Enrollment', 'Update'], len(df), p=[0.7, 0.3])
        if 'Update_Type' not in df.columns:
            df['Update_Type'] = np.where(df['Enrollment_Type'] == 'Update', 
                                       np.random.choice(['Address', 'Mobile', 'Biometric', 'Name', 'DOB'], len(df)),
                                       'Not Applicable')
        if 'Update_Reason' not in df.columns:
            df['Update_Reason'] = np.where(df['Enrollment_Type'] == 'Update',
                                         np.random.choice(['Change of Address', 'Lost Card', 'Damaged Card', 'Data Correction'], len(df)),
                                         'Not Specified')
        if 'Status' not in df.columns:
            df['Status'] = np.random.choice(
                ['Completed', 'In Progress', 'Pending Verification', 'Rejected'], 
                len(df), 
                p=[0.85, 0.08, 0.05, 0.02]
            )
        
        # Handle missing values
        df.fillna({
            'State': 'Unknown',
            'Gender': 'Unknown',
            'Age': int(df['Age'].median()) if 'Age' in df and not df['Age'].isnull().all() else 30,
            'Update_Type': 'Not Applicable',
            'Update_Reason': 'Not Specified',
            'Enrollment_Type': 'New Enrollment'
        }, inplace=True)
        
        # Ensure all required columns are present
        required_cols = ['Enrollment_Date', 'Year', 'Month', 'Day_Of_Week', 'Hour', 'State', 'Gender', 'Age', 'Enrollment_Type']
        for col in required_cols:
            if col not in df.columns:
                st.warning(f"Warning: Missing column {col} in data. Creating default values.")
                if col == 'Year':
                    df[col] = df['Enrollment_Date'].dt.year
                elif col == 'Month':
                    df[col] = df['Enrollment_Date'].dt.month
                elif col == 'Day_Of_Week':
                    df[col] = df['Enrollment_Date'].dt.dayofweek
                elif col == 'Hour':
                    df[col] = df['Enrollment_Date'].dt.hour
                elif col in ['State', 'Gender']:
                    df[col] = col
                elif col == 'Age':
                    df[col] = np.random.normal(35, 15, len(df)).clip(1, 100).astype(int)
                elif col == 'Enrollment_Type':
                    df[col] = np.random.choice(['New Enrollment', 'Update'], len(df), p=[0.7, 0.3])
        
        return df
        
    except FileNotFoundError:
        st.warning("Data file not found. Using sample data for demonstration.")
        return generate_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}\nUsing sample data instead.")
        return generate_sample_data()

def analyze_trends(df):
    """Analyze enrollment and update trends with anomaly detection"""
    # Time series decomposition
    df_daily = df.set_index('Enrollment_Date')['Aadhaar_Number'].resample('D').count()
    
    # Handle missing dates by filling with 0
    idx = pd.date_range(df_daily.index.min(), df_daily.index.max())
    df_daily = df_daily.reindex(idx, fill_value=0)
    
    # Decompose time series
    try:
        decomposition = seasonal_decompose(df_daily, period=30, extrapolate_trend='freq')
        
        # Detect anomalies using z-score on residuals
        residual = decomposition.resid.dropna()
        z_scores = np.abs(stats.zscore(residual))
        anomalies = residual[z_scores > 3]
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': residual,
            'anomalies': anomalies
        }
    except Exception as e:
        st.warning(f"Could not perform time series analysis: {str(e)}")
        return None

def generate_insights(df):
    """Generate actionable insights from the data"""
    insights = []
    
    # 1. Overall enrollment status
    if 'Status' in df.columns:
        status_dist = df['Status'].value_counts(normalize=True) * 100
        status_insight = (
            f"{status_dist.get('Completed', 0):.1f}% of enrollments are completed. "
            f"{status_dist.get('Rejected', 0):.1f}% were rejected."
        )
        insights.append(status_insight)
    else:
        insights.append("Status information not available in the dataset.")
    
    # 2. Peak enrollment times
    if 'Hour_Of_Day' in df.columns:
        peak_hour = df['Hour_Of_Day'].mode()[0]
        insights.append(f"Peak enrollment hour is {peak_hour}:00.")
    
    # 3. Most common update type
    if 'Update_Type' in df.columns and df['Update_Type'].notna().any():
        common_update = df['Update_Type'].mode()[0]
        insights.append(f"Most common update type: {common_update}.")
    
    # 4. State with highest enrollment
    if 'State' in df.columns:
        top_state = df['State'].value_counts().idxmax()
        insights.append(f"{top_state} has the highest number of enrollments.")
    
    # 5. Age group distribution
    if 'Age_Group' in df.columns:
        top_age_group = df['Age_Group'].value_counts().idxmax()
        insights.append(f"Most common age group: {top_age_group} years.")
    
    # 6. Processing time
    if 'Processing_Time_Minutes' in df.columns:
        avg_processing = df['Processing_Time_Minutes'].mean()
        insights.append(f"Average processing time: {avg_processing:.1f} minutes.")
    
    # 7. Enrollment type distribution
    if 'Enrollment_Type' in df.columns:
        new_vs_update = df['Enrollment_Type'].value_counts(normalize=True) * 100
        insights.append(
            f"Enrollment types: {new_vs_update.get('New Enrollment', 0):.1f}% new, "
            f"{new_vs_update.get('Update', 0):.1f}% updates."
        )
    
    # 8. Weekend vs weekday pattern
    if 'Weekend' in df.columns:
        weekend_pct = df['Weekend'].mean() * 100
        insights.append(f"{weekend_pct:.1f}% of enrollments happen on weekends.")
    
    # 9. Seasonal pattern
    if 'Update_Season' in df.columns:
        season_dist = df['Update_Season'].value_counts(normalize=True) * 100
        insights.append(
            f"Enrollment distribution by season: {season_dist.get('High', 0):.1f}% high, "
            f"{season_dist.get('Medium', 0):.1f}% medium, "
            f"{season_dist.get('Low', 0):.1f}% low."
        )
    
    # 10. Gender distribution
    if 'Gender' in df.columns:
        gender_dist = df['Gender'].value_counts(normalize=True) * 100
        insights.append(
            f"Gender distribution: {gender_dist.get('Male', 0):.1f}% male, "
            f"{gender_dist.get('Female', 0):.1f}% female, "
            f"{gender_dist.get('Other', 0):.1f}% other."
        )
    
    return insights[:5]  # Return top 5 insights

def create_geospatial_map(df):
    """Create an interactive map showing enrollment distribution"""
    # Indian states with coordinates for mapping
    states_info = {
        'Maharashtra': {'lat': 19.7515, 'lon': 75.7139, 'population': 123144223},
        'Uttar Pradesh': {'lat': 26.8467, 'lon': 80.9462, 'population': 199812341},
        'Karnataka': {'lat': 15.3173, 'lon': 75.7139, 'population': 61095297},
        'Tamil Nadu': {'lat': 11.1271, 'lon': 78.6569, 'population': 72147030},
        'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'population': 16787941},
        'West Bengal': {'lat': 22.9868, 'lon': 87.8550, 'population': 91276115},
        'Gujarat': {'lat': 22.2587, 'lon': 71.1924, 'population': 60439692},
        'Rajasthan': {'lat': 27.0238, 'lon': 74.2179, 'population': 68548437},
        'Bihar': {'lat': 25.0961, 'lon': 85.3131, 'population': 104099452},
        'Andhra Pradesh': {'lat': 15.9129, 'lon': 79.7400, 'population': 49577103},
        'Madhya Pradesh': {'lat': 22.9734, 'lon': 78.6569, 'population': 72626809},
        'Kerala': {'lat': 10.8505, 'lon': 76.2711, 'population': 33406061},
        'Punjab': {'lat': 31.1471, 'lon': 75.3412, 'population': 27743338},
        'Haryana': {'lat': 29.0588, 'lon': 76.0856, 'population': 25351462},
        'Odisha': {'lat': 20.9517, 'lon': 85.0985, 'population': 41974218},
        'Unknown': {'lat': 20.5937, 'lon': 78.9629, 'population': 0}  # Default center of India
    }
    
    # Create a base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='cartodbpositron')
    
    # Add circle markers for each enrollment
    sample_size = min(1000, len(df))
    if sample_size == 0:
        st.warning("No data available for the selected filters.")
        return m
        
    for idx, row in df.sample(sample_size).iterrows():
        state = row.get('State', 'Unknown')
        
        # Get coordinates - first try direct columns, then states_info, then default to center of India
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            lat, lon = row['Latitude'], row['Longitude']
        else:
            state_info = states_info.get(state, states_info['Unknown'])
            lat = state_info['lat'] + np.random.normal(0, 0.5)  # Add some jitter
            lon = state_info['lon'] + np.random.normal(0, 0.5)
        
        # Ensure coordinates are valid numbers
        try:
            lat = float(lat)
            lon = float(lon)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Coordinates out of range")
        except (ValueError, TypeError):
            # Fallback to state center if coordinates are invalid
            state_info = states_info.get(state, states_info['Unknown'])
            lat, lon = state_info['lat'], state_info['lon']
        
        # Create popup text
        popup_text = f"<b>State:</b> {state}<br>"
        if 'Enrollment_Type' in df.columns:
            popup_text += f"<b>Type:</b> {row['Enrollment_Type']}<br>"
        if 'Status' in df.columns:
            popup_text += f"<b>Status:</b> {row['Status']}"
        
        # Add marker to map
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=state
        ).add_to(m)
    
    # Add a choropleth layer if we have state data
    try:
        state_counts = df['State'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        
        # Create a GeoJSON layer for Indian states (simplified)
        # In a production environment, you would load a proper GeoJSON file here
        # This is a simplified version for demonstration
        geojson = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': {'name': state},
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [states_info.get(state, states_info['Unknown'])['lon'], 
                                      states_info.get(state, states_info['Unknown'])['lat']]
                    }
                }
                for state in state_counts['State'].unique()
            ]
        }
        
        # Add the GeoJSON layer to the map
        folium.GeoJson(
            geojson,
            name='State Boundaries',
            style_function=lambda x: {
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.1
            }
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
    except Exception as e:
        st.warning(f"Could not create state boundaries layer: {str(e)}")
    
    return m

def predict_future_enrollments(df, periods=12):
    """Predict future enrollment trends using Prophet"""
    try:
        # Prepare data for time series forecasting
        df_ts = df.resample('M', on='Enrollment_Date').size().reset_index()
        df_ts.columns = ['ds', 'y']
        
        # Train the model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(df_ts)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        # Plot the forecast
        fig = model.plot(forecast)
        plt.title('Enrollment Forecast')
        plt.xlabel('Date')
        plt.ylabel('Number of Enrollments')
        
        # Add components to the plot
        fig2 = model.plot_components(forecast)
        
        return fig, fig2
    except Exception as e:
        st.warning(f"Could not generate forecast: {str(e)}")
        return None, None

def main():
    # Responsive header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", 
                width=80, use_column_width=False)
    with col2:
        st.markdown("<h1 class='main-header'>Aadhaar Analytics Dashboard</h1>", 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Add a mobile menu button
    st.markdown("""<button class="mobile-menu-btn" onclick="document.querySelector('[data-testid=stSidebar]').classList.toggle('sidebar-visible')">☰</button>""", 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Add age groups
    age_bins = [0, 18, 30, 45, 60, 100]
    age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # Add day names
    df['Day_Name'] = df['Enrollment_Date'].dt.day_name()
    
    # Sidebar filters with responsive design
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        # Responsive filter layout
        col1, col2 = st.columns(2)
        
        # Year filter
        years = sorted(df['Year'].unique(), reverse=True)
        with col1:
            selected_years = st.multiselect(
                'Select Year(s)',
                options=years,
                default=[max(years)] if years else None,
                key='year_selector'
            )
            
        # State filter
        states = ['All'] + sorted(df['State'].unique().tolist())
        with col2:
            selected_state = st.selectbox(
                'Select State',
                options=states,
                index=0,
                key='state_selector'
            )
            
        # Enrollment type filter
        enrollment_types = ['All'] + df['Enrollment_Type'].unique().tolist()
        with col1:
            selected_enrollment_type = st.selectbox(
                'Enrollment Type',
                options=enrollment_types,
                index=0,
                key='enrollment_type_selector'
            )
            
        # Age group filter
        with col2:
            age_groups = ['All', '0-18', '19-35', '36-50', '51-65', '65+']
            selected_age_group = st.selectbox(
                'Age Group',
                options=age_groups,
                index=0,
                key='age_group_selector'
            )
            
        # Add a divider
        st.markdown("---")
        
        # Add a refresh button
        if st.button('🔄 Apply Filters', use_container_width=True):
            st.experimental_rerun()
            
        # Add some space at the bottom for mobile
        st.markdown("<div style='margin-bottom: 80px;'></div>", unsafe_allow_html=True)
    
    # Apply filters
    df_filtered = df.copy()
    
    # Apply year filter
    if selected_years and 'All' not in selected_years:
        df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
    
    # Apply state filter
    if selected_state != 'All':
        df_filtered = df_filtered[df_filtered['State'] == selected_state]
    
    # Apply enrollment type filter
    if selected_enrollment_type != 'All':
        df_filtered = df_filtered[df_filtered['Enrollment_Type'] == selected_enrollment_type]
    
    # Apply age group filter
    if selected_age_group != 'All':
        if selected_age_group == '0-18':
            df_filtered = df_filtered[df_filtered['Age'] <= 18]
        elif selected_age_group == '19-35':
            df_filtered = df_filtered[(df_filtered['Age'] >= 19) & (df_filtered['Age'] <= 35)]
        elif selected_age_group == '36-50':
            df_filtered = df_filtered[(df_filtered['Age'] >= 36) & (df_filtered['Age'] <= 50)]
        elif selected_age_group == '51-65':
            df_filtered = df_filtered[(df_filtered['Age'] >= 51) & (df_filtered['Age'] <= 65)]
        elif selected_age_group == '65+':
            df_filtered = df_filtered[df_filtered['Age'] > 65]
    
    # Display loading spinner while analyzing
    with st.spinner('Analyzing data...'):
        # Generate insights
        insights = generate_insights(df_filtered)
        
        # Analyze trends
        trend_analysis = analyze_trends(df_filtered)
    
    # Responsive KPI Cards
    kpi_cols = st.columns([1, 1, 1, 1])
    
    with kpi_cols[0]:
        st.metric("📊 Total", 
                 f"{len(df_filtered):,}",
                 help="Total number of records")
    
    with kpi_cols[1]:
        new_enrollments = len(df_filtered[df_filtered['Enrollment_Type'] == 'New Enrollment'])
        st.metric("🆕 New Enrollments", 
                 f"{new_enrollments:,}",
                 help="Number of new Aadhaar enrollments")
    
    with kpi_cols[2]:
        updates = len(df_filtered[df_filtered['Enrollment_Type'] == 'Update'])
        st.metric("🔄 Updates", 
                 f"{updates:,}",
                 help="Number of Aadhaar updates")
    
    with kpi_cols[3]:
        avg_age = df_filtered['Age'].mean()
        st.metric("👥 Avg Age", 
                 f"{avg_age:.1f} years",
                 help="Average age of applicants")
    
    # Add a small gap after KPIs
    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    
    # Display insights
    if insights:
        st.markdown("### 🔍 Key Insights")
        for i, insight in enumerate(insights, 1):
            with st.expander(f"Insight #{i}"):
                st.write(insight)
                # Add a general recommendation based on the insight
                if 'completed' in insight.lower() and 'rejected' in insight.lower():
                    st.info("💡 Recommendation: Monitor the rejection rate and investigate any unusual patterns.")
                elif 'peak' in insight.lower() and 'hour' in insight.lower():
                    st.info("💡 Recommendation: Consider allocating more resources during peak hours to handle the load.")
                elif 'common update type' in insight.lower():
                    st.info("💡 Recommendation: Focus on optimizing the most common update type to improve efficiency.")
                else:
                    st.info("💡 Recommendation: Review this insight for potential process improvements.")
    
    # Trend analysis section
    st.markdown("### 📈 Trend Analysis")
    
    # Check if we have enough data for trend analysis (at least 24 months of data)
    min_date = df_filtered['Enrollment_Date'].min()
    max_date = df_filtered['Enrollment_Date'].max()
    months_of_data = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
    
    if months_of_data < 24:
        st.warning(
            f"⚠️ Limited data for trend analysis"
            f"\n• You have {months_of_data} month{'' if months_of_data == 1 else 's'} of data (from {min_date.strftime('%b %Y')} to {max_date.strftime('%b %Y')})"
            "\n• For more accurate trend detection, it's recommended to have at least 24 months of data"
            "\n• Current analysis may not show meaningful patterns due to limited historical data"
        )
        
        # Show a progress bar for data coverage
        coverage_pct = min(100, int((months_of_data / 24) * 100))
        st.progress(coverage_pct, text=f"Data coverage: {coverage_pct}% of recommended minimum")
        
        # Show tips for getting better analysis
        with st.expander("📊 How to improve your analysis"):
            st.markdown("""
            To get more accurate trend analysis:
            1. Select a larger date range if available
            2. Use the sample data option to see a complete analysis
            3. Check back when you have more historical data
            """)
    
    # Proceed with analysis if we have any data
    if len(df_filtered) > 0 and trend_analysis and not trend_analysis['anomalies'].empty:
        # Create a dataframe for the plot
        plot_df = pd.DataFrame({
            'Date': trend_analysis['trend'].index,
            'Trend': trend_analysis['trend'].values,
            'Anomaly': trend_analysis['trend'].index.isin(trend_analysis['anomalies'].index)
        })
        
        # Create the plot
        fig = px.line(plot_df, x='Date', y='Trend', title='Enrollment Trend with Anomalies')
        
        # Add anomaly markers
        anomalies_df = plot_df[plot_df['Anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalies_df['Date'],
            y=anomalies_df['Trend'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Anomaly'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show anomaly details
        with st.expander("View Anomaly Details"):
            st.write("The following dates showed unusual activity:")
            for date, value in trend_analysis['anomalies'].items():
                st.write(f"- {date.date()}: {int(value):,} enrollments (deviation from trend)")
    else:
        st.warning("Insufficient data for trend analysis. Try expanding your date range.")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🗺️ Geospatial", "🔮 Predictive Analytics"])
    
    with tab1:
        # Overview Tab - Show key metrics and trends
        st.markdown("### Enrollment Trends Over Time")
        
        # Time series chart
        fig = px.line(
            df_filtered.groupby(['Enrollment_Date']).size().reset_index(name='Count'),
            x='Enrollment_Date',
            y='Count',
            title='Daily Enrollment/Update Count',
            labels={'Enrollment_Date': 'Date', 'Count': 'Number of Records'}
        )
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Add more overview visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Enrollment by Type
            fig = px.pie(
                df_filtered, 
                names='Enrollment_Type', 
                title='Enrollment Type Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Enrollment by Status
            fig = px.bar(
                df_filtered['Status'].value_counts().reset_index(),
                x='Status',
                y='count',
                title='Enrollment Status Distribution',
                labels={'count': 'Count', 'Status': 'Status'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Geospatial Tab
        st.markdown("### Enrollment Distribution by Location")
        
        # Create and display the map
        m = create_geospatial_map(df_filtered)
        folium_static(m, width=1000, height=600)
        
        # State-wise enrollment counts
        st.markdown("### State-wise Enrollment Summary")
        state_summary = df_filtered['State'].value_counts().reset_index()
        state_summary.columns = ['State', 'Enrollment Count']
        st.dataframe(
            state_summary,
            use_container_width=True,
            height=300
        )
    
    with tab3:
        # Predictive Analytics Tab
        st.markdown("### Enrollment Forecast")
        
        # Add a slider for forecast period
        forecast_months = st.slider(
            "Select number of months to forecast:",
            min_value=1,
            max_value=24,
            value=12,
            step=1
        )
        
        # Generate and display forecast
        with st.spinner('Generating forecast...'):
            forecast_fig, components_fig = predict_future_enrollments(df_filtered, forecast_months)
            
            if forecast_fig is not None:
                st.pyplot(forecast_fig)
                st.markdown("### Forecast Components")
                st.pyplot(components_fig)
                
                # Add some interpretation
                st.markdown("#### How to interpret the forecast:")
                st.markdown("""
                - **Trend**: Shows the overall direction of enrollment numbers
                - **Yearly**: Displays seasonal patterns that repeat every year
                - **Weekly**: Shows day-of-week patterns in the data
                - **Residuals**: The difference between observed and predicted values
                """)
            else:
                st.warning("Could not generate forecast. Please ensure you have sufficient historical data (at least 6 months).")
    
    # Row 2: Demographics
    st.markdown("### Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            df_filtered, 
            x='Age', 
            nbins=30, 
            title='Age Distribution',
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = df_filtered['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig_gender = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender', 
            title='Gender Distribution',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Advanced Analysis Section
    st.markdown("---")
    st.markdown("## 🔍 Advanced Analysis")
    
    # Anomaly Detection
    st.markdown("### 🚨 Anomaly Detection")
    
    # Prepare data for anomaly detection
    daily_counts = df_filtered.groupby('Enrollment_Date').size().reset_index(name='Count')
    daily_counts['z_score'] = stats.zscore(daily_counts['Count'])
    anomalies = daily_counts[(daily_counts['z_score'] > 3) | (daily_counts['z_score'] < -3)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly visualization
        fig_anomaly = px.scatter(
            daily_counts,
            x='Enrollment_Date',
            y='Count',
            title='Enrollment Anomalies',
            color=np.where(
                (daily_counts['z_score'] > 3) | (daily_counts['z_score'] < -3),
                'Anomaly',
                'Normal'
            ),
            labels={'color': 'Status'},
            color_discrete_map={
                'Normal': '#1E88E5',
                'Anomaly': '#FF5252'
            }
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        # Anomaly details
        if not anomalies.empty:
            st.markdown("#### Detected Anomalies")
            for idx, row in anomalies.iterrows():
                st.warning(f"📅 {row['Enrollment_Date'].strftime('%Y-%m-%d')}: {int(row['Count'])} enrollments (Z-score: {row['z_score']:.2f})")
            
            # Possible reasons for anomalies
            st.markdown("#### Possible Reasons")
            st.markdown("""
            - Government initiatives or awareness campaigns
            - Technical issues in data collection
            - Special enrollment drives
            - System updates or maintenance periods
            """)
        else:
            st.success("✅ No significant anomalies detected in the selected time period.")
    
    # Trend Analysis
    st.markdown("### 📈 Trend Analysis")
    
    # Time series decomposition
    try:
        # Resample data to monthly frequency and fill any missing months with 0
        ts_data = df_filtered.set_index('Enrollment_Date').resample('M').size()
        
        # Ensure we have at least 24 months of data for meaningful decomposition
        if len(ts_data) < 24:
            st.warning("⚠️ At least 24 months of data are recommended for accurate trend analysis.")
            return None
            
        # Perform seasonal decomposition with robust error handling
        try:
            decomposition = seasonal_decompose(ts_data, period=12, extrapolate_trend='freq')
        except Exception as e:
            st.warning(f"⚠️ Could not perform seasonal decomposition: {str(e)}")
            # Fall back to simple trend analysis if seasonal decomposition fails
            decomposition = seasonal_decompose(ts_data, model='additive', period=1, extrapolate_trend='freq')
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend,
            name='Trend',
            line=dict(color='#1E88E5', width=2)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=decomposition.seasonal.index,
            y=decomposition.seasonal + decomposition.trend,
            name='Seasonal',
            line=dict(color='#FF9800', width=1, dash='dot')
        ))
        
        fig_trend.update_layout(
            title='Time Series Decomposition',
            xaxis_title='Date',
            yaxis_title='Enrollment Count',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Trend insights
        trend_slope = np.polyfit(range(len(ts_data)), ts_data, 1)[0]
        if trend_slope > 0:
            st.success(f"📈 Upward trend detected: Enrollment is increasing over time")
        elif trend_slope < 0:
            st.warning(f"📉 Downward trend detected: Enrollment is decreasing over time")
        else:
            st.info("➡️ Stable trend: No significant increase or decrease in enrollments")
            
    except Exception as e:
        st.error(f"Could not perform trend analysis: {str(e)}")
    
    # Demographic Insights
    st.markdown("### 👥 Demographic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age group analysis
        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
        filtered_df['Age_Group'] = pd.cut(
            filtered_df['Age'], 
            bins=age_bins, 
            labels=age_labels,
            right=False
        )
        
        age_group_counts = filtered_df['Age_Group'].value_counts().sort_index()
        fig_age_group = px.bar(
            age_group_counts,
            x=age_group_counts.index,
            y=age_group_counts.values,
            title='Enrollments by Age Group',
            labels={'x': 'Age Group', 'y': 'Count'},
            color_discrete_sequence=['#4CAF50']
        )
        st.plotly_chart(fig_age_group, use_container_width=True)
    
    with col2:
        # Gender and age heatmap
        if 'Gender' in filtered_df.columns:
            gender_age = pd.crosstab(
                filtered_df['Gender'],
                filtered_df['Age_Group'],
                normalize='index'  # Normalize by row (gender)
            )
            
            fig_heatmap = px.imshow(
                gender_age,
                labels=dict(x="Age Group", y="Gender", color="Proportion"),
                x=gender_age.columns,
                y=gender_age.index,
                aspect="auto",
                color_continuous_scale='Blues',
                title='Age Distribution by Gender'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # State to ISO code mapping
    state_to_iso = {
        'Andhra Pradesh': 'IN-AP',
        'Arunachal Pradesh': 'IN-AR',
        'Assam': 'IN-AS',
        'Bihar': 'IN-BR',
        'Chhattisgarh': 'IN-CT',
        'Goa': 'IN-GA',
        'Gujarat': 'IN-GJ',
        'Haryana': 'IN-HR',
        'Himachal Pradesh': 'IN-HP',
        'Jharkhand': 'IN-JH',
        'Karnataka': 'IN-KA',
        'Kerala': 'IN-KL',
        'Madhya Pradesh': 'IN-MP',
        'Maharashtra': 'IN-MH',
        'Manipur': 'IN-MN',
        'Meghalaya': 'IN-ML',
        'Mizoram': 'IN-MZ',
        'Nagaland': 'IN-NL',
        'Odisha': 'IN-OR',
        'Punjab': 'IN-PB',
        'Rajasthan': 'IN-RJ',
        'Sikkim': 'IN-SK',
        'Tamil Nadu': 'IN-TN',
        'Telangana': 'IN-TG',
        'Tripura': 'IN-TR',
        'Uttar Pradesh': 'IN-UP',
        'Uttarakhand': 'IN-UT',
        'West Bengal': 'IN-WB',
        'Unknown': 'IN-AP'  # Default for any unknown states
    }
    
    state_counts = filtered_df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    
    # Add ISO codes to the dataframe
    state_counts['ISO'] = state_counts['State'].map(state_to_iso)
    
    # Create the choropleth map
    fig_state = px.choropleth(
        state_counts,
        locations='ISO',
        color='Count',
        hover_name='State',
        color_continuous_scale='Blues',
        title='Enrollments by State',
        scope='asia',
        locationmode='ISO-3',
        projection='mercator'
    )
    fig_state.update_geos(
        visible=False, 
        projection_scale=5, 
        center={"lat": 20.5937, "lon": 78.9629},  # Center on India
        showcountries=True,
        countrycolor="Black"
    )
    st.plotly_chart(fig_state, use_container_width=True)
    
    # Row 4: Update Analysis (only show if updates are selected)
    if selected_enrollment_type in ['Update', 'All'] and 'Update' in filtered_df['Enrollment_Type'].unique():
        st.markdown("### Update Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Update types
            update_types = filtered_df[filtered_df['Enrollment_Type'] == 'Update']['Update_Type'].value_counts()
            fig_upd_type = px.bar(
                update_types,
                title='Types of Updates',
                labels={'index': 'Update Type', 'value': 'Count'}
            )
            st.plotly_chart(fig_upd_type, use_container_width=True)
        
        with col2:
            # Update reasons
            update_reasons = filtered_df[filtered_df['Enrollment_Type'] == 'Update']['Update_Reason'].value_counts()
            fig_upd_reason = px.pie(
                update_reasons,
                names=update_reasons.index,
                values=update_reasons.values,
                title='Update Reasons',
                hole=0.4
            )
            st.plotly_chart(fig_upd_reason, use_container_width=True)
    
    # Row 5: Time-based patterns
    st.markdown("### Time-based Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly pattern
        monthly = filtered_df.groupby('Month').size().reset_index(name='Count')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly['Month'] = monthly['Month'].apply(lambda x: months[x-1])
        
        fig_monthly = px.line(
            monthly,
            x='Month',
            y='Count',
            title='Monthly Enrollment Pattern',
            markers=True
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Daily pattern
        hourly = filtered_df.groupby('Hour_Of_Day').size().reset_index(name='Count')
        
        fig_hourly = px.bar(
            hourly,
            x='Hour_Of_Day',
            y='Count',
            title='Hourly Enrollment Pattern',
            labels={'Hour_Of_Day': 'Hour of Day (24h)'}
        )
        fig_hourly.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Aadhaar Analytics Dashboard**  
    This dashboard provides insights into Aadhaar enrollment and update patterns.  
    Data is synthetic and for demonstration purposes only.
    """)

if __name__ == "__main__":
    main()
