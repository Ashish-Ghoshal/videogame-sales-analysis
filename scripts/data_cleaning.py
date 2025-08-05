import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Loads the dataset from the specified CSV file path.

    Args:
        file_path (str): The complete path to the raw CSV data file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}. Please ensure the file is in the correct location.")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Performs data cleaning and preprocessing steps on the DataFrame.
    This includes handling missing values, correcting data types, and
    engineering new features.

    Args:
        df (pd.DataFrame): The raw DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned and processed DataFrame.
    """
    if df is None:
        return None

    print("\n--- Starting Data Cleaning and Preprocessing ---")

    # Drop rows with missing 'Genre' or 'Publisher' as these are critical categorical features
    initial_rows = df.shape[0]
    df.dropna(subset=['Genre', 'Publisher'], inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} rows with missing 'Genre' or 'Publisher'.")

    # Handle missing 'Year_of_Release': Fill with mode or drop. Dropping for simplicity here.
    initial_rows = df.shape[0]
    df.dropna(subset=['Year_of_Release'], inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} rows with missing 'Year_of_Release'.")

    # Convert 'Year_of_Release' to integer
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)

    # Convert 'User_Score' to numeric first, coercing errors to NaN (e.g., 'tbd' values)
    # This must happen before attempting to calculate median or fill NaNs for User_Score.
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    print("Converted 'User_Score' to numeric, coercing non-convertible values to NaN.")

    # Handle missing 'Critic_Score' and 'User_Score'
    # Fill with the median to be robust to outliers.
    for col in ['Critic_Score', 'User_Score']:
        if df[col].isnull().any():
            median_val = df[col].median()
            # Assign the result back to the column to avoid SettingWithCopyWarning
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing '{col}' with median value: {median_val}")

    # Feature Engineering: Create a simple average score if both exist
    # Ensure scores are numeric before calculation
    df['Avg_Score'] = (df['Critic_Score'] + df['User_Score'] * 10) / 20
    print("Engineered 'Avg_Score' feature.")

    # Drop any duplicate rows based on 'Name', 'Platform', 'Year_of_Release'
    initial_rows = df.shape[0]
    df.drop_duplicates(subset=['Name', 'Platform', 'Year_of_Release'], inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")

    print(f"Data cleaning complete. New shape: {df.shape}")
    return df

def save_data(df, file_path):
    """
    Saves the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The complete path where the CSV file will be saved.
    """
    if df is None:
        print("No DataFrame to save.")
        return

    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    # Define paths
    RAW_DATA_PATH = os.path.join('data', 'raw', 'video_game_sales_and_ratings.csv')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_data.csv')

    # Load, clean, and save data
    raw_df = load_data(RAW_DATA_PATH)
    if raw_df is not None:
        # Use .copy() to ensure operations on cleaned_df don't affect raw_df
        cleaned_df = clean_data(raw_df.copy())
        save_data(cleaned_df, PROCESSED_DATA_PATH)
