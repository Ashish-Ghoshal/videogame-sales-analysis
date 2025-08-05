import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_processed_data(file_path):
    """
    Loads the processed dataset from the specified CSV file path.

    Args:
        file_path (str): The complete path to the processed CSV data file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: Processed data not found at {file_path}. Please run data_cleaning.py first.")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded processed data for model training from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading processed data for model training: {e}")
        return None

def preprocess_for_model(df):
    """
    Preprocesses the DataFrame for machine learning model training.
    This includes handling categorical features using Label Encoding and
    selecting relevant features.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Features (X)
            - pd.Series: Target variable (y)
            - dict: Dictionary of LabelEncoders used for categorical columns.
    """
    if df is None:
        return None, None, None

    print("\n--- Preprocessing Data for Model Training ---")

    # Drop rows with any remaining NaNs that might affect model training
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} rows with remaining NaNs for model training.")

    # Select features and target
    # Exclude 'Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales' as they are either identifiers
    # or directly related to the target 'Global_Sales' (which is the sum of regional sales).
    # 'Rank' is also an identifier.
    features = [
        'Platform', 'Year_of_Release', 'Genre', 'Publisher',
        'Critic_Score', 'User_Score', 'Avg_Score'
    ]
    target = 'Global_Sales'

    X = df[features].copy()
    y = df[target]

    # Apply Label Encoding to categorical features
    categorical_cols = ['Platform', 'Genre', 'Publisher']
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            print(f"Applied Label Encoding to '{col}'.")
        else:
            print(f"Warning: Categorical column '{col}' not found in features.")

    print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    return X, y, label_encoders

def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        sklearn.ensemble.RandomForestRegressor: The trained model.
    """
    print("\n--- Training RandomForestRegressor Model ---")
    # Initialize the Random Forest Regressor
    # Using default parameters for simplicity; hyperparameter tuning can be added later.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints performance metrics.

    Args:
        model (sklearn.ensemble.RandomForestRegressor): The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Basic interpretation of R2
    if r2 > 0.75:
        print("R-squared indicates a strong fit, explaining a large proportion of variance in sales.")
    elif r2 > 0.5:
        print("R-squared indicates a moderate fit, explaining a fair amount of variance in sales.")
    else:
        print("R-squared indicates a weak fit, suggesting the model may not capture sales variance well.")


def save_model(model, encoders, model_path='models/game_sales_model.pkl', encoders_path='models/label_encoders.pkl'):
    """
    Saves the trained model and label encoders to disk.

    Args:
        model (sklearn.ensemble.RandomForestRegressor): The trained model.
        encoders (dict): Dictionary of LabelEncoders.
        model_path (str): Path to save the model.
        encoders_path (str): Path to save the label encoders.
    """
    models_dir = os.path.dirname(model_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    try:
        joblib.dump(model, model_path)
        print(f"Trained model saved to {model_path}")
        joblib.dump(encoders, encoders_path)
        print(f"Label encoders saved to {encoders_path}")
    except Exception as e:
        print(f"Error saving model or encoders: {e}")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_data.csv')
    MODEL_PATH = os.path.join('models', 'game_sales_model.pkl')
    ENCODERS_PATH = os.path.join('models', 'label_encoders.pkl')

    df = load_processed_data(PROCESSED_DATA_PATH)
    if df is not None:
        X, y, label_encoders = preprocess_for_model(df)
        if X is not None and y is not None:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

            model = train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)
            save_model(model, label_encoders, MODEL_PATH, ENCODERS_PATH)
