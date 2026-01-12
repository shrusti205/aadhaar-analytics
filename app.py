import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Aadhaar Analytics Dashboard",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size: 36px; color: #1E88E5; margin-bottom: 20px;}
    .section-header {font-size: 24px; color: #1E88E5; margin: 20px 0 10px 0;}
    .metric-card {background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px 0;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .stSelectbox, .stSlider, .stDateInput {margin-bottom: 15px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    try:
        # Try to load the data from CSV
        df = pd.read_csv("aadhaar_data.csv")
        df['Enrollment_Date'] = pd.to_datetime(df['Enrollment_Date'])
        df['Year'] = df['Enrollment_Date'].dt.year
        df['Month'] = df['Enrollment_Date'].dt.month
        df['Day_Of_Week'] = df['Enrollment_Date'].dt.dayofweek
        return df
    except FileNotFoundError:
        st.warning("Data file not found. Please run 'python data_generator.py' first to generate the data.")
        return None

def main():
    st.markdown("<h1 class='main-header'>Aadhaar Enrollment & Update Analytics</h1>", unsafe_allow_html=True)
    st.markdown("""
    This dashboard provides insights into Aadhaar enrollment and update patterns across different 
    regions and demographics. Use the filters below to explore the data.
    """)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Year filter
    years = sorted(df['Year'].unique(), reverse=True)
    selected_years = st.sidebar.multiselect(
        'Select Years',
        options=years,
        default=[2023, 2022, 2021]
    )
    
    # State filter
    states = ['All'] + sorted(df['State'].unique().tolist())
    selected_state = st.sidebar.selectbox('Select State', states, index=0)
    
    # Enrollment type filter
    enrollment_types = ['All'] + df['Enrollment_Type'].unique().tolist()
    selected_enrollment_type = st.sidebar.selectbox('Enrollment Type', enrollment_types, index=0)
    
    # Apply filters
    filtered_df = df[df['Year'].isin(selected_years)]
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    
    if selected_enrollment_type != 'All':
        filtered_df = filtered_df[filtered_df['Enrollment_Type'] == selected_enrollment_type]
    
    # KPI Cards
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Enrollments", f"{len(filtered_df):,}")
    
    with col2:
        new_enrollments = len(filtered_df[filtered_df['Enrollment_Type'] == 'New Enrollment'])
        st.metric("New Enrollments", f"{new_enrollments:,}")
    
    with col3:
        updates = len(filtered_df[filtered_df['Enrollment_Type'] == 'Update'])
        st.metric("Updates Performed", f"{updates:,}")
    
    with col4:
        avg_age = filtered_df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    # Row 1: Time Series and Map
    st.markdown("### Enrollment Trends Over Time")
    
    # Time series chart
    fig = px.line(
        filtered_df.groupby(['Enrollment_Date']).size().reset_index(name='Count'),
        x='Enrollment_Date',
        y='Count',
        title='Daily Enrollment/Update Count',
        labels={'Enrollment_Date': 'Date', 'Count': 'Number of Records'}
    )
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Demographics
    st.markdown("### Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            filtered_df, 
            x='Age', 
            nbins=30, 
            title='Age Distribution',
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = filtered_df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig_gender = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender', 
            title='Gender Distribution',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Row 3: Geographic Analysis
    st.markdown("### Geographic Distribution")
    
    # State-wise distribution
    state_counts = filtered_df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    
    fig_state = px.choropleth(
        state_counts,
        locationmode='country names',
        locations=state_counts['State'],
        color='Count',
        hover_name='State',
        color_continuous_scale='Blues',
        title='Enrollments by State',
        scope='asia',
        locationmode='country names'
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
