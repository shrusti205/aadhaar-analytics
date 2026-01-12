# Aadhaar Analytics Dashboard

A comprehensive dashboard for analyzing Aadhaar enrollment and update patterns across different regions and demographics in India.

## Features

- Interactive visualizations of enrollment trends over time
- Demographic analysis by age and gender
- Geographic distribution of enrollments
- Analysis of update types and reasons
- Time-based patterns (monthly, daily, hourly)
- Interactive filters for year, state, and enrollment type

## Setup Instructions

1. **Prerequisites**
   - Python 3.8 or higher
   - pip (Python package installer)

2. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd aadhaar-analytics
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Generate synthetic data**
   ```bash
   python data_generator.py
   ```
   This will create a `aadhaar_data.csv` file with 50,000 synthetic records.

6. **Run the Streamlit app locally**
   ```bash
   streamlit run app.py
   ```
   The app will open in your default web browser at `http://localhost:8501`

## Project Structure

- `app.py`: Main Streamlit application
- `data_generator.py`: Script to generate synthetic Aadhaar data
- `aadhaar_data.csv`: Generated synthetic data (created after running data_generator.py)
- `requirements.txt`: Python dependencies

## Deployment to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and connect your GitHub repository
4. Select the main branch and set the main file to `app.py`
5. Click "Deploy!"

## Data Privacy

This application uses synthetic data that mimics Aadhaar enrollment patterns. No real Aadhaar data is used in this application.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
