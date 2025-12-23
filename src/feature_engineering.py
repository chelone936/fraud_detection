import pandas as pd
import numpy as np

def create_time_features(df):
    """Creates time-based features from purchase_time and signup_time."""
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    return df

def create_velocity_features(df):
    """Creates transaction frequency and velocity features."""
    # Since we observed mostly 1 transaction per user/device in this specific dataset (EDA insight), 
    # we target device usage and IP reuse as proxies for velocity.
    df['device_usage_count'] = df.groupby('device_id')['device_id'].transform('count')
    df['ip_usage_count'] = df.groupby('ip_address')['ip_address'].transform('count')
    return df
