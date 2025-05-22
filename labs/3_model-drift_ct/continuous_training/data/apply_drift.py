#!/usr/bin/env python3
import pandas as pd
import numpy as np
import datetime
import argparse

# Used to generate drifted data for testing the Azure Designer Pipeline

#example usage : python apply_drift.py --input outputs/prepared_data.csv
#                                      --output outputs/drifted_data.csv

def create_drift_wine_data(original_data, drift_percentage=0.3, magnitude=0.5):
    """
    Create a drifted version of the wine dataset.
    
    Args:
        original_data: Original wine dataset
        drift_percentage: Percentage of features to apply drift to
        magnitude: Magnitude of the drift
    
    Returns:
        DataFrame with drifted values
    """
    # Make a copy of the original data
    drift_data = original_data.copy()
    
    # Select a subset of features to drift (excluding target and timestamp)
    features = [col for col in drift_data.columns if col not in ['quality', 'good_quality', 'timestamp']]
    num_features_to_drift = max(1, int(len(features) * drift_percentage))
    features_to_drift = np.random.choice(features, num_features_to_drift, replace=False)
    
    # Apply drift to selected features
    for feature in features_to_drift:
        # Get the standard deviation of the feature
        std_dev = drift_data[feature].std()
        
        # Apply a shift in mean and increase in variance
        drift_data[feature] = drift_data[feature] + np.random.normal(
            loc=magnitude * std_dev,  # Shift mean
            scale=magnitude * std_dev,  # Increase variance
            size=len(drift_data)
        )
    
    # Create a newer timestamp if it exists in the data
    if 'timestamp' in drift_data.columns:
        latest_date = original_data['timestamp'].max()
        new_dates = [latest_date + datetime.timedelta(days=i+1) for i in range(len(drift_data))]
        drift_data['timestamp'] = new_dates
    
    print(f"Drift applied to features: {', '.join(features_to_drift)}")
    return drift_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply drift to wine quality dataset')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--drift-percentage', type=float, default=0.4, 
                        help='Percentage of features to apply drift to (default: 0.4)')
    parser.add_argument('--magnitude', type=float, default=0.8, 
                        help='Magnitude of the drift (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Read input data
    print(f"Reading data from {args.input}")
    df = pd.read_csv(args.input, sep=',')
    
    # Handle timestamp column if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply drift to the data
    print(f"Applying drift with percentage={args.drift_percentage}, magnitude={args.magnitude}")
    drift_data = create_drift_wine_data(
        df, 
        drift_percentage=args.drift_percentage, 
        magnitude=args.magnitude
    )
    
    # Save the drifted data
    print(f"Saving drifted data to {args.output}")
    drift_data.to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()