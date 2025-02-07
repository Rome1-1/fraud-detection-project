import pandas as pd
import numpy as np
import ipaddress
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data():
    """Loads datasets and handles errors if files are missing."""
    try:
        print("Loading data...")
        fraud_data = pd.read_csv("data/Fraud_Data.csv")
        ip_data = pd.read_csv("data/IpAddress_to_Country.csv")
        creditcard_data = pd.read_csv("data/creditcard.csv")

        # Debugging: Print dataset shapes
        print(f"Fraud data loaded: {fraud_data.shape}")
        print(f"IP data loaded: {ip_data.shape}")
        print(f"Credit card data loaded: {creditcard_data.shape}")
        
        print("Data loaded successfully.")
        return fraud_data, ip_data, creditcard_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

def handle_missing_values(df, name="dataset"):
    """Handles missing values by filling with median and dropping remaining nulls."""
    print(f"Handling missing values in {name}... Initial shape: {df.shape}")
    
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(inplace=True)

    print(f"Shape after handling missing values: {df.shape}")
    print(f"Missing values left in {name}: {df.isnull().sum().sum()}")
    return df

def remove_duplicates(df, name="dataset"):
    """Removes duplicate rows."""
    print(f"Removing duplicates in {name}... Initial shape: {df.shape}")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows. New shape: {df.shape}")
    return df

def convert_timestamps(df):
    """Converts time columns to datetime format."""
    print("Converting timestamps...")
    for col in ["signup_time", "purchase_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    print("Timestamps converted.")
    return df

def convert_scientific_to_ip(ip):
    """Converts scientific notation to standard IP format."""
    try:
        ip_int = int(float(ip))  # Convert from scientific to integer
        return ipaddress.IPv4Address(ip_int)
    except ValueError:
        return None

test_ips = ["7.327584e+08", "3.503114e+08", "2.621474e+09", "3.840542e+09", "4.155831e+08"]
test_ips_converted = [convert_scientific_to_ip(ip) for ip in test_ips]
print(test_ips_converted)

def merge_ip_data(fraud_data, ip_data):
    """Merges fraud data with IP address country data, handling NaNs properly."""
    print("Merging IP data...")

    # Check first few rows of the fraud data for valid IP addresses
    print(f"Fraud data IP addresses before conversion:\n{fraud_data['ip_address'].head()}")
    
    # Convert IP addresses safely
    fraud_data["ip_address"] = fraud_data["ip_address"].apply(convert_scientific_to_ip)  # Use the correct function here
    ip_data["lower_bound_ip_address"] = ip_data["lower_bound_ip_address"].apply(convert_scientific_to_ip)
    ip_data["upper_bound_ip_address"] = ip_data["upper_bound_ip_address"].apply(convert_scientific_to_ip)

    # Print a few samples after conversion to check the result
    print("Fraud data IP addresses after conversion:\n", fraud_data['ip_address'].head())
    print("Sample lower bound IPs:", ip_data["lower_bound_ip_address"].head())
    print("Sample upper bound IPs:", ip_data["upper_bound_ip_address"].head())

    # Remove rows with invalid IP addresses
    fraud_data = fraud_data.dropna(subset=["ip_address"])
    if fraud_data["ip_address"].isna().sum() > 0:
        print(f"Warning: Found {fraud_data['ip_address'].isna().sum()} invalid IP addresses in fraud data.")
    
    # Continue with the merge if data is valid
    if fraud_data.empty:
        print("Warning: No valid IP addresses in fraud dataset after conversion.")
        return fraud_data

    merged_data = fraud_data.merge(ip_data, how="left", left_on="ip_address", right_on="lower_bound_ip_address")
    
    if merged_data["country"].isna().sum() > 0:
        merged_data = merged_data.merge(ip_data, how="left", left_on="ip_address", right_on="upper_bound_ip_address", suffixes=("", "_alt"))
        merged_data["country"].fillna(merged_data["country_alt"], inplace=True)
        merged_data.drop(columns=["country_alt"], errors="ignore", inplace=True)

    merged_data.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"], errors="ignore", inplace=True)

    print(f"Merged data size after IP join: {merged_data.shape}")
    return merged_data




def feature_engineering(df):
    """Adds transaction-based and time-based features."""
    print("Performing feature engineering...")
    if df.empty:
        print("No data available for feature engineering.")
        return df
    df["transaction_count"] = df.groupby("user_id")["user_id"].transform("count")
    df["device_transaction_count"] = df.groupby("device_id")["device_id"].transform("count")
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    print("Feature engineering completed.")
    return df

def scale_features(df, feature_cols):
    """Scales numerical features if data is available."""
    print(f"Scaling features: {feature_cols}...")
    if df.empty:
        print("No data available for scaling.")
        return df
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"Features scaled: {feature_cols}")
    return df

def encode_categorical(df, cat_cols):
    """Encodes categorical columns with Label Encoding."""
    print(f"Encoding categorical columns: {cat_cols}...")
    if df.empty:
        print("No data available for encoding.")
        return df
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))
    print(f"Categorical columns encoded: {cat_cols}")
    return df

def preprocess_and_save():
    """Runs full preprocessing pipeline and saves cleaned data."""
    print("Starting preprocessing...")
    fraud_data, ip_data, creditcard_data = load_data()
    
    fraud_data = handle_missing_values(fraud_data, "Fraud Data")
    creditcard_data = handle_missing_values(creditcard_data, "Credit Card Data")
    
    fraud_data = remove_duplicates(fraud_data, "Fraud Data")
    creditcard_data = remove_duplicates(creditcard_data, "Credit Card Data")
    
    fraud_data = convert_timestamps(fraud_data)
    fraud_data = merge_ip_data(fraud_data, ip_data)
    
    if not fraud_data.empty:
        fraud_data = feature_engineering(fraud_data)
        fraud_data = scale_features(fraud_data, ["purchase_value"])
        fraud_data = encode_categorical(fraud_data, ["source", "browser", "country"])
    else:
        print("No fraud data to process after IP merge.")
    
    creditcard_data = scale_features(creditcard_data, ["Amount"])
    
    # Ensure output directory exists
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print("Saving cleaned data...")
    if not fraud_data.empty:
        fraud_data.to_csv(f"{output_dir}/Fraud_Data_Cleaned.csv", index=False)
    else:
        print("Warning: No fraud data available to save.")

    if not creditcard_data.empty:
        creditcard_data.to_csv(f"{output_dir}/CreditCard_Cleaned.csv", index=False)
    else:
        print("Warning: No credit card data available to save.")

    print("Preprocessing completed. Cleaned data saved.")

if __name__ == "__main__":
    preprocess_and_save()
