import pandas as pd
import numpy as np

def load_data(fraud_path, ip_mapping_path):
    """Loads the fraud and ip mapping datasets."""
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_mapping_path)
    return fraud_df, ip_df

def clean_data(df):
    """Performs basic data cleaning."""
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df = df.drop_duplicates()
    return df

def map_ip_to_country(fraud_df, ip_df):
    """Merges fraud data with ip mapping using range-based lookup."""
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    merged_df = pd.merge_asof(
        fraud_df,
        ip_df,
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    mask = (merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) & \
           (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])
    
    merged_df.loc[~mask, 'country'] = 'Unknown'
    merged_df = merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
    
    return merged_df
