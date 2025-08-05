import subprocess
import os
import sys

def run_script(script_path):
    """
    Executes a Python script using subprocess.

    Args:
        script_path (str): The complete path to the Python script to be executed.
    """
    print(f"\n--- Running {script_path} ---")
    try:
        # Use sys.executable to ensure the script is run with the same Python interpreter
        # that is executing this main.py script. This is crucial for virtual environments.
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors/Warnings from {script_path}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(f"Command: {e.cmd}")
        print(f"Return Code: {e.returncode}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1) # Exit if a script fails
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}. Please check the path.")
        sys.exit(1)

def main():
    """
    Orchestrates the execution of the video game sales analysis workflow.
    This script sequentially calls the data cleaning, EDA, model training,
    and prediction scripts. It also provides an option to skip model training
    if a model already exists.
    """
    print("Starting Video Game Sales Analysis Workflow...")

    # Define paths to scripts relative to the main.py location
    scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    data_cleaning_script = os.path.join(scripts_dir, 'data_cleaning.py')
    eda_script = os.path.join(scripts_dir, 'eda.py')
    train_model_script = os.path.join(scripts_dir, 'train_model.py')
    predict_sales_script = os.path.join(scripts_dir, 'predict_sales.py')

    # Define paths for model and encoders
    model_path = os.path.join('models', 'game_sales_model.pkl')
    encoders_path = os.path.join('models', 'label_encoders.pkl')

    # Ensure necessary directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('plots/static', exist_ok=True)
    os.makedirs('plots/html', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Execute data cleaning and EDA scripts
    run_script(data_cleaning_script)
    run_script(eda_script)

    # Check if model exists and prompt user
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        while True:
            user_input = input("\nExisting model found. Do you want to retrain the model? (yes/no): ").lower().strip()
            if user_input in ['yes', 'y']:
                print("Retraining model...")
                run_script(train_model_script)
                run_script(predict_sales_script)
                break
            elif user_input in ['no', 'n']:
                print("Skipping model retraining. Using existing model for prediction.")
                run_script(predict_sales_script) # Still run predict_sales with existing model
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
    else:
        print("\nNo existing model found. Training a new model.")
        run_script(train_model_script)
        run_script(predict_sales_script)

    print("\nVideo Game Sales Analysis Workflow Completed.")
    print("Please check the 'plots/static/' directory for static visualizations,")
    print("the 'plots/html/' directory for interactive plots, and open 'plots/html/dashboard.html'")
    print("in your web browser for the interactive dashboard.")
    print("The trained model is saved in 'models/game_sales_model.pkl'.")

if __name__ == "__main__":
    main()
