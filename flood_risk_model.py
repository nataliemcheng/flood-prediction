# flood_risk_model.py
#THis is our code from the jupyter notebook rewritten as a python script for testing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import RFE

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    df_encoded = pd.get_dummies(data, columns=["Land Cover", "Soil Type"], drop_first=True)
    
    X = df_encoded.drop(columns=["Flood Occurred"])
    y = df_encoded["Flood Occurred"]

    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    scaler = StandardScaler()
    X_resampled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X.columns)

    return X_resampled, y_resampled

def feature_selection(X_resampled, y_resampled):
    rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
    X_selected = rfe.fit_transform(X_resampled, y_resampled)
    
    selected_features = X_resampled.columns[rfe.support_]
    return X_selected, selected_features

def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    return log_reg

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, roc_auc
