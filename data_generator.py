import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_aadhaar_data(num_records=10000):
    """Generate synthetic Aadhaar enrollment and update data."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate base data
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    # States and districts in India
    states = ["Maharashtra", "Uttar Pradesh", "Bihar", "West Bengal", "Madhya Pradesh", 
              "Tamil Nadu", "Rajasthan", "Karnataka", "Gujarat", "Andhra Pradesh"]
    districts = {
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Meerut"],
        "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"],
        "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri"],
        "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Kota", "Bikaner", "Ajmer"],
        "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Gulbarga"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
        "Andhra Pradesh": ["Hyderabad", "Visakhapatnam", "Vijayawada", "Guntur", "Nellore"]
    }
    
    # Generate random data
    data = []
    
    for _ in range(num_records):
        # Random date between 2010 and 2023
        days_offset = random.randint(0, date_range)
        enrollment_date = start_date + timedelta(days=days_offset)
        
        # Random state and district
        state = random.choice(states)
        district = random.choice(districts[state])
        
        # Age distribution (weighted towards younger ages)
        age = int(np.random.beta(2, 5) * 100) + 1
        
        # Gender distribution (slightly more males in India)
        gender = np.random.choice(["Male", "Female"], p=[0.52, 0.48])
        
        # Enrollment type (new or update)
        if random.random() < 0.7:  # 70% new enrollments
            enrollment_type = "New Enrollment"
            update_type = "N/A"
            update_reason = "N/A"
        else:  # 30% updates
            enrollment_type = "Update"
            update_type = random.choice(["Address Update", "Biometric Update", "Demographic Update", "All Updates"])
            update_reason = random.choice(["Change of Address", "Biometric Mismatch", 
                                         "Name Change", "Date of Birth Correction"])
        
        # Generate a random Aadhaar-like number (12 digits)
        aadhaar_number = ''.join([str(random.randint(0, 9)) for _ in range(12)])
        
        # Generate random enrollment/update time (to the nearest 15 minutes)
        enrollment_time = enrollment_date + timedelta(
            hours=random.randint(0, 23),
            minutes=15 * random.randint(0, 3)
        )
        
        data.append({
            "Aadhaar_Number": aadhaar_number,
            "Enrollment_Type": enrollment_type,
            "Enrollment_Date": enrollment_date.strftime("%Y-%m-%d"),
            "Enrollment_Time": enrollment_time.strftime("%H:%M"),
            "Update_Type": update_type,
            "Update_Reason": update_reason,
            "State": state,
            "District": district,
            "Age": age,
            "Gender": gender,
            "Year": enrollment_date.year,
            "Month": enrollment_date.month,
            "Day_Of_Week": enrollment_date.weekday(),  # 0 is Monday, 6 is Sunday
            "Hour_Of_Day": enrollment_time.hour
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values (2% of the data)
    for col in df.columns:
        if col not in ["Aadhaar_Number", "Enrollment_Date", "Enrollment_Time"]:
            mask = np.random.random(len(df)) < 0.02
            df.loc[mask, col] = np.nan
    
    return df

if __name__ == "__main__":
    # Generate and save the data
    print("Generating synthetic Aadhaar data...")
    df = generate_synthetic_aadhaar_data(50000)  # Generate 50,000 records
    df.to_csv("aadhaar_data.csv", index=False)
    print("Data generated and saved to 'aadhaar_data.csv'")
    print(f"Total records: {len(df)}")
