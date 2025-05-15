
#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# python .\components\drift_detection\drift_detection.py    --historical_data .\outputs\prepared_data.csv 
#                                                           --current_data .\outputs\drifted_data.csv 
#                                                           --output_path .\outputs\retraining_needed.json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical_data", type=str, help="Path to historical data CSV file")
    parser.add_argument("--current_data", type=str, help="Path to current data CSV file")
    parser.add_argument("--output_path", type=str, help="Path to output training needed result JSON file")
    return parser.parse_args()

def prepare_discriminator_data(original_data, new_data):
    """
    Prepare data for training a discriminator model to detect drift.
    
    Args:
        original_data: Original dataset
        new_data: New dataset that may have drift
        
    Returns:
        X: Combined features from both datasets
        y: Binary labels (0 for original, 1 for new)
    """
    # Get features only - exclude target and timestamp
    original_features = original_data.drop(['quality', 'good_quality', 'timestamp'], axis=1, errors='ignore')
    new_features = new_data.drop(['quality', 'good_quality', 'timestamp'], axis=1, errors='ignore')
    
    # Label the data sources: 0 for original, 1 for new
    original_features['source'] = 0
    new_features['source'] = 1
    
    # Combine the datasets
    combined_data = pd.concat([original_features, new_features], axis=0)
    
    # Shuffle the data for better training
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split features and label
    X = combined_data.drop('source', axis=1)
    y = combined_data['source']
    
    return X, y

def main():
    args = parse_args()
    
    # Load datasets
    print("Loading historical and current datasets...")
    historical_data = pd.read_csv(args.historical_data)
    current_data = pd.read_csv(args.current_data)
    
    # Prepare data for the discriminator
    X_disc, y_disc = prepare_discriminator_data(historical_data, current_data)
    X_disc_train, X_disc_test, y_disc_train, y_disc_test = train_test_split(X_disc, y_disc, test_size=0.2, random_state=42)
    
    # Train the discriminator model
    print("Training discriminator model...")
    discriminator_model = RandomForestClassifier(n_estimators=100, random_state=42)
    discriminator_model.fit(X_disc_train, y_disc_train)

    # Evaluate feature importance
    importances = discriminator_model.feature_importances_
    feature_names = X_disc_train.columns  # Si X_disc_train est un DataFrame

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("Evaluated feature importance for bias detection. The higher the number, the more biased the data.")
    print(feature_importance_df)
    
    # Evaluate the discriminator
    y_disc_pred_proba = discriminator_model.predict_proba(X_disc_test)[:, 1]
    discriminator_auc = roc_auc_score(y_disc_test, y_disc_pred_proba)
    print(f"Discriminator AUC-ROC: {discriminator_auc:.4f}")
    
    # Determine if training is needed based on discriminator performance
    # If AUC-ROC > 0.7, the model can distinguish between datasets,
    # indicating significant drift that requires retraining
    training_needed = discriminator_auc > 0.7
    if training_needed:
        print("\n" + "="*50)
        print("🚨 RETRAINING NEEDED: Significant drift detected!")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("✅ NO RETRAINING NEEDED: No significant drift detected.")
        print("="*50 + "\n")
    
    # Save the result
    with open(args.output_path, "w") as f:
        json.dump({"training_needed": bool(training_needed)}, f)

if __name__ == "__main__":
    main()