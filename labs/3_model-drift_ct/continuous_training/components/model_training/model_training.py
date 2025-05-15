#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#exemple usage : python model_training.py --input_data data/winequality_cleaned.csv 
#                                            --model_output model/wine_quality_model.pt
#                                            --metrics_output metrics/wine_quality_metrics.json
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to input data CSV file")
    parser.add_argument("--model_output", type=str, help="Path to output model file (.pt)")
    parser.add_argument("--metrics_output", type=str, help="Path to output metrics file (.json)")
    return parser.parse_args()

def main():
    # Parse args
    args = parse_args()
    
    # Read in the wine quality dataset directly from file path
    print("Loading data...")
    wine_data = pd.read_csv(args.input_data)
    
    # Prepare data for training - extract features and target
    print("Preparing data for training...")
    X = wine_data.drop(['good_quality', 'timestamp'], axis=1, errors='ignore')
    y = wine_data['good_quality']  # Binary classification target
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    # Save the model directly to the output file path
    print("Saving model...")
    joblib.dump({
        'model': model,
        'feature_names': X.columns.tolist()
    }, args.model_output)
    
    # Save metrics directly to the output file path
    print("Saving metrics...")
    metrics = {"f1_score": float(f1)}
    
    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f)
    
    print("Done!")

if __name__ == "__main__":
    main()