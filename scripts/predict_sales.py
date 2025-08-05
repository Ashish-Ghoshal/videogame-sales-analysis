import pandas as pd
import joblib
import os
import numpy as np

def load_model_and_encoders(model_path, encoders_path):
    """
    Loads the trained machine learning model and label encoders from disk.

    Args:
        model_path (str): Path to the saved model file.
        encoders_path (str): Path to the saved label encoders file.

    Returns:
        tuple: A tuple containing the loaded model and encoders, or (None, None) if an error occurs.
    """
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        encoders = joblib.load(encoders_path)
        print(f"Successfully loaded label encoders from {encoders_path}")
        return model, encoders
    except FileNotFoundError:
        print(f"Error: Model or encoders file not found. Please ensure '{model_path}' and '{encoders_path}' exist.")
        print("Run 'python main.py' or 'python scripts/train_model.py' to train and save the model.")
        return None, None
    except Exception as e:
        print(f"Error loading model or encoders: {e}")
        return None, None

def make_prediction(model, encoders, input_data):
    """
    Makes a sales prediction for a single game based on input features.

    Args:
        model: The trained machine learning model.
        encoders (dict): Dictionary of LabelEncoders used during training.
        input_data (dict): A dictionary containing the features for a single game.
                           Example: {'Platform': 'PS2', 'Year_of_Release': 2004,
                                     'Genre': 'Action', 'Publisher': 'Activision',
                                     'Critic_Score': 85.0, 'User_Score': 8.0,
                                     'Avg_Score': 82.5}

    Returns:
        float: The predicted Global_Sales, or None if prediction fails.
    """
    if model is None or encoders is None:
        return None

    print("\n--- Making a Sales Prediction ---")

    # Convert input_data to a DataFrame, ensuring column order matches training data
    # The order of features must be consistent with how the model was trained.
    # Refer to `train_model.py` for the feature list.
    features_order = [
        'Platform', 'Year_of_Release', 'Genre', 'Publisher',
        'Critic_Score', 'User_Score', 'Avg_Score'
    ]
    input_df = pd.DataFrame([input_data], columns=features_order)

    # Apply the same Label Encoding as during training
    for col, le in encoders.items():
        if col in input_df.columns:
            # Handle unseen labels by transforming them to a default value (e.g., -1 or the most frequent label)
            # For simplicity, if an unseen label is encountered, it will raise an error unless handled.
            # A robust solution would involve a custom transformer or more sophisticated handling.
            # Here, we'll try to transform and catch errors.
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError as e:
                print(f"Warning: Unseen label for '{col}' detected. Error: {e}")
                print(f"Input value: {input_data[col]}. Consider adding this label to training data or handling it specifically.")
                # For demonstration, we'll set it to a known value or drop the prediction
                # In a real app, you might map it to a 'unknown' category or use one-hot encoding.
                return None # Abort prediction if an unseen label is critical

    try:
        prediction = model.predict(input_df)[0]
        print(f"Predicted Global Sales: {prediction:.2f} Million Units")
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    MODEL_PATH = os.path.join('models', 'game_sales_model.pkl')
    ENCODERS_PATH = os.path.join('models', 'label_encoders.pkl')

    model, encoders = load_model_and_encoders(MODEL_PATH, ENCODERS_PATH)

    if model and encoders:
        # Example input for prediction
        sample_game_data = {
            'Platform': 'PS2',
            'Year_of_Release': 2004,
            'Genre': 'Action',
            'Publisher': 'Activision',
            'Critic_Score': 85.0,
            'User_Score': 8.0,
            'Avg_Score': (85.0 + 8.0 * 10) / 20 # Calculate Avg_Score consistent with data_cleaning.py
        }
        print("\nSample Game Data for Prediction:")
        for key, value in sample_game_data.items():
            print(f"- {key}: {value}")

        predicted_sales = make_prediction(model, encoders, sample_game_data)
        if predicted_sales is not None:
            print(f"\nPrediction successful for the sample game.")
        else:
            print("\nPrediction failed for the sample game. Check warnings/errors above.")

        # Another example: a hypothetical new game
        new_game_data = {
            'Platform': 'PS5', # Assuming PS5 would be handled by encoder if it was in training data
            'Year_of_Release': 2023,
            'Genre': 'Sports',
            'Publisher': 'EA Sports',
            'Critic_Score': 92.0,
            'User_Score': 8.5,
            'Avg_Score': (92.0 + 8.5 * 10) / 20
        }
        print("\nAnother Sample Game Data for Prediction (Hypothetical):")
        for key, value in new_game_data.items():
            print(f"- {key}: {value}")
        predicted_sales_new = make_prediction(model, encoders, new_game_data)
        if predicted_sales_new is not None:
            print(f"\nPrediction successful for the hypothetical new game.")
        else:
            print("\nPrediction failed for the hypothetical new game. This might happen if new categorical values (like 'PS5') are not in the trained encoder.")
