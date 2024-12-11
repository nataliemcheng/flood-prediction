# test_flood_risk_model.py

import pytest
import pandas as pd
from flood_risk_model import preprocess_data, feature_selection, train_logistic_regression, evaluate_model

def test_preprocess_data(tmpdir):
    # Create a temporary CSV file
    test_csv = tmpdir.join("test_data.csv")

    # Create a mock DataFrame with all required columns
    mock_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0],
        'Land Cover': ['forest', 'desert', 'grassland'],
        'Soil Type': ['clay', 'sand', 'loam'],
        'Flood Occurred': [1, 0, 1]  # Include this column
    })

    # Save the DataFrame to the temporary file
    mock_data.to_csv(test_csv, index=False)

    # Call your preprocessing function
    X_resampled, y_resampled = preprocess_data(str(test_csv))

    # Assertions for the output
    assert X_resampled is not None
    assert y_resampled is not None

def test_feature_selection():
    X_resampled = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9],
    })
    y_resampled = [0, 1, 0]
    
    # Specify n_features=2 to match the test case
    X_selected, selected_features = feature_selection(X_resampled, y_resampled, n_features=2)
    
    assert len(selected_features) == 2   # Should select 2 features based on RFE

def test_train_logistic_regression():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = [0, 1, 0]
    model = train_logistic_regression(X_train, y_train)
    assert model is not None

def test_evaluate_model():
    from sklearn.model_selection import train_test_split
    
    # Ensure consistent lengths for features and labels
    X_resampled = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [4, 5, 6, 7, 8],
    })
    y_resampled = [0, 1, 0, 1, 0]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )
    
    # Train the model
    model = train_logistic_regression(X_train, y_train)
    
    # Evaluate the model
    accuracy, roc_auc = evaluate_model(model, X_test, y_test)
    
    # Assert expected values (replace with your expectations)
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= roc_auc <= 1.0
  # ROC AUC should also be valid
