# test_flood_risk_model.py

import pytest
import pandas as pd
from flood_risk_model import preprocess_data, feature_selection, train_logistic_regression, evaluate_model

def test_preprocess_data():
    # Mock file path or create a small test dataframe for testing
    X_resampled, y_resampled = preprocess_data('test_data.csv')
    assert X_resampled.shape[0] == y_resampled.shape[0]
    assert len(X_resampled.columns) > 0

def test_feature_selection():
    X_resampled = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9],
    })
    y_resampled = [0, 1, 0]
    
    X_selected, selected_features = feature_selection(X_resampled, y_resampled)
    assert len(selected_features) == 2  # Should select 2 features based on RFE

def test_train_logistic_regression():
    # Test that the logistic regression model can be trained
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = [0, 1, 0]
    model = train_logistic_regression(X_train, y_train)
    assert model is not None

def test_evaluate_model():
    # Test evaluation function
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_resampled = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
    })
    y_resampled = [0, 1, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)
    model = train_logistic_regression(X_train, y_train)
    accuracy, roc_auc = evaluate_model(model, X_test, y_test)
    
    assert accuracy >= 0  # Accuracy should be a valid number
    assert roc_auc >= 0  # ROC AUC should also be valid