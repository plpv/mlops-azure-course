import os
import argparse
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt

# example usage : python data_preparation.py --input data/winequality-red.csv 
#                                            --prepared_data prepared_data/winequality_cleaned.csv 
# 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--prepared_data', type=str, required=True)
    args = parser.parse_args()

    # Resolve paths relative to the root of the project
    root_dir = os.path.abspath(os.path.dirname(__file__))  # Directory of this script
    prepared_data_path = os.path.join(root_dir, args.prepared_data)

    # Load and clean data
    df = pd.read_csv(args.input, sep=',')
    
    # 1. Remove rows with any null values
    df_clean = df.dropna()
    
    # 2. Create binary quality classification
    df_clean['good_quality'] = (df_clean['quality'] >= 6).astype(int)
    df_clean.drop(columns=['quality'], inplace=True)    

    print(f"Data after cleaning: {df_clean.shape}")
    print(df_clean.head())
    
    # 3. Save cleaned data as CSV
    print(os.path.dirname(args.prepared_data))
    os.makedirs(os.path.dirname(prepared_data_path), exist_ok=True)

    df_clean.to_csv(args.prepared_data, index=False)
    
if __name__ == '__main__':
    main()