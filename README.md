# Video Game Sales Analysis & Prediction

## Project Overview

This repository contains a comprehensive data science project focused on analyzing the dynamics of the video game industry through sales and ratings data. The primary objective is to gain a deep understanding of factors influencing game success, identify trends across platforms and genres, and build a predictive model for global sales.

Leveraging the "Video Game Sales and Ratings" dataset, this project aims to:

*   **Uncover Data Characteristics:** Understand distributions of sales, ratings, release years, and game attributes.
    
*   **Identify Relationships:** Explore correlations between game features (e.g., genre, platform, critic score) and sales performance.
    
*   **Build a Predictive Model:** Develop a machine learning model capable of predicting a game's `Global_Sales` based on various attributes.
    

This repository serves as a demonstration of a structured, reproducible, and insightful approach to a data science problem, with a strong emphasis on data understanding and visualization, designed to stand out.

## Features

*   **Robust Data Cleaning & Preprocessing:** Scripts to handle missing values, convert data types, and prepare raw data for analysis and modeling. This includes processing numerical scores and categorical attributes like genre and platform.
    
*   **Comprehensive Exploratory Data Analysis (EDA):**
    
    *   Generation of detailed statistical summaries.
        
    *   Creation of a diverse set of static visualizations (histograms, bar plots, correlation heatmaps) using Matplotlib and Seaborn.
        
*   **Interactive Visualizations:** Production of dynamic, interactive plots using Plotly, saved as standalone HTML files, allowing for deeper data exploration (zooming, panning, hovering for details on sales trends, genre performance, and score comparisons).
    
*   **Interactive Mini-Dashboard:** A single HTML file (`dashboard.html`) that aggregates key interactive plots, offering a concise and engaging overview of the dataset's insights into industry trends and game performance.
    
*   **Feature Engineering:** Derivation of new, insightful features (e.g., total review count, average score if applicable).
    
*   **Predictive Modeling:** Implementation of a machine learning pipeline to train a regression model (Random Forest Regressor) for predicting `Global_Sales`.
    
*   **Model Evaluation:** Thorough assessment of the trained model's performance using standard regression metrics (e.g., Mean Absolute Error, R-squared).
    
*   **Modular and Reproducible Codebase:** A well-organized directory structure with distinct scripts for each stage of the data science workflow, promoting maintainability and ease of replication.
    

## Technologies Used

*   **Python 3.9+**
    
*   **Pandas:** For efficient data manipulation and analysis.
    
*   **NumPy:** For fundamental numerical operations.
    
*   **Scikit-learn:** For machine learning model building, training, and evaluation.
    
*   **Matplotlib & Seaborn:** For static, high-quality data visualizations.
    
*   **Plotly:** For creating interactive and dynamic data visualizations, including the mini-dashboard.
    
*   **Joblib:** For efficient serialization and deserialization of Python objects (used for saving/loading the trained model).
    
*   **Tailwind CSS:** (Used in `dashboard.html`) A utility-first CSS framework for rapid and responsive UI development.
    

## File and Directory Structure

    .
    ├── data/
    │   ├── raw/
    │   │   └── video_game_sales_and_ratings.csv  # Raw dataset (download separately)
    │   └── processed/
    │       └── processed_data.csv   # Cleaned and featurized data
    ├── plots/
    │   ├── static/
    │   │   ├── global_sales_distribution.png
    │   │   ├── year_of_release_distribution.png
    │   │   ├── regional_sales_distribution.png
    │   │   ├── sales_by_platform_static.png
    │   │   └── top_genres_static.png
    │   ├── html/
    │   │   ├── sales_by_genre_interactive.html
    │   │   ├── sales_trends_interactive.html
    │   │   ├── critic_vs_user_score_interactive.html
    │   │   ├── regional_sales_by_genre_interactive.html
    │   │   ├── regional_sales_trends_interactive.html
    │   │   ├── sales_by_platform_interactive.html
    │   │   └── dashboard.html  # Interactive mini-dashboard
    │   └── correlation_heatmap.png # Static heatmap
    ├── scripts/
    │   ├── data_cleaning.py
    │   ├── eda.py
    │   ├── train_model.py
    │   └── predict_sales.py
    ├── models/
    │   └── game_sales_model.pkl    # Trained machine learning model
    ├── .gitignore
    ├── requirements.txt
    ├── main.py                     # Orchestrates the entire workflow
    └── README.md
    
    

## Getting Started

### Prerequisites

Ensure you have **Python 3.9 or higher** installed on your system.

### Installation

1.  **Clone the repository:**
    
        git clone https://github.com/Ashish-Ghoshal/videogame-sales-analysis.git
        cd videogame-sales-analysis
        
    
2.  **Create a unique virtual environment:** It's highly recommended to use a virtual environment to manage project dependencies.
    
        python -m venv vid_venv
        
    
3.  **Activate the virtual environment:**
    
    *   **On macOS/Linux:**
        
            source vid_venv/bin/activate
            
        
    *   **On Windows (Command Prompt):**
        
            vid_venv\Scripts\activate.bat
            
        
    *   **On Windows (PowerShell):**
        
            .\vid_venv\Scripts\Activate.ps1
            
        
4.  **Install dependencies:**
    
        pip install -r requirements.txt
        
    
5.  **Download the dataset:**
    
    *   Navigate to the [Video Game Sales and Ratings dataset on Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales "null").
        
    *   Download the `video_game_sales_and_ratings.csv` file.
        
    *   **Crucially, place the downloaded file into the `data/raw/` directory** within your cloned repository.
        

### Usage

The project workflow is sequential. You can run the entire pipeline using the `main.py` script, or execute individual scripts as needed.

1.  **Run the entire workflow:**
    
        python main.py
        
    
    This command will:
    
    *   Process the raw data (`data_cleaning.py`).
        
    *   Perform exploratory data analysis and generate plots (`eda.py`).
        
    *   Train the predictive model (`train_model.py`).
        
    *   Demonstrate a sample prediction (`predict_sales.py`).
        
2.  **View the Interactive Dashboard:** After running `eda.py` (either via `main.py` or directly), open the `dashboard.html` file in your web browser. This file is located at `plots/html/dashboard.html`.
    
        # Example (replace with your actual path or open manually)
        your_browser_command plots/html/dashboard.html
        
    
3.  **Deactivate the virtual environment (when done):**
    
        deactivate
        
    

## Interpreting the Results & Insights

### Static Plots (`plots/static/`)

*   `global_sales_distribution.png`: This histogram illustrates the distribution of `Global_Sales` (in millions).
    
    *   **Insight:** Often highly skewed, showing that most games have relatively low sales, with a long tail for blockbuster titles. This highlights the challenge of predicting hits.
        
*   `year_of_release_distribution.png`: A histogram showing the distribution of games by their `Year_of_Release`.
    
    *   **Insight:** Reveals periods of high game releases, indicating industry growth or specific console generations.
        
*   `regional_sales_distribution.png`: A set of bar plots showing the total sales for North America, Europe, and Japan.
    
    *   **Insight:** Provides a quick overview of which regions are the largest markets in terms of total sales volume. This helps in understanding market dominance.
        
*   `sales_by_platform_static.png`: A bar plot showing the total `Global_Sales` for the top N platforms.
    
    *   **Insight:** Identifies which gaming platforms have generated the most revenue over time, indicating platform market leadership.
        
*   `top_genres_static.png`: A bar plot showing the total `Global_Sales` for the top N genres.
    
    *   **Insight:** Identifies the most commercially successful genres in the video game market.
        
*   `correlation_heatmap.png`: A heatmap displaying the Pearson correlation coefficients between all numerical features.
    
    *   **Insight:** Identifies strong relationships between features like sales figures across different regions, and between sales and critic/user scores. This helps in understanding which factors are most indicative of overall sales.
        

### Interactive Plots (`plots/html/`) and Dashboard (`dashboard.html`)

These interactive plots, accessible via the dashboard, offer deeper insights and help answer specific business questions:

*   `regional_sales_by_genre_interactive.html` (within dashboard): An interactive grouped bar chart showing `NA_Sales`, `EU_Sales`, and `JP_Sales` aggregated by Genre.
    
    *   **Insight:** This plot directly addresses "Are certain genres disproportionately popular in specific regions?" You can easily compare the sales performance of Action games in North America versus Japan, or Sports games in Europe. This helps understand regional market preferences.
        
*   `regional_sales_trends_interactive.html` (within dashboard): An interactive line plot showing `NA_Sales`, `EU_Sales`, and `JP_Sales` trends over `Year_of_Release`.
    
    *   **Insight:** This visualizes "How have regional sales trends evolved over time compared to global sales?" You can observe if certain regions experienced booms or busts independently, or if global trends are driven by specific regional markets. It helps identify market maturity and growth phases per region.
        
*   `sales_by_platform_interactive.html` (within dashboard): An interactive bar chart showing total `Global_Sales` by Platform.
    
    *   **Insight:** Allows for precise comparison of total sales across all platforms, with hover details for exact figures. It helps identify "Which platforms have dominated sales over time?" and see the overall market share.
        
*   `sales_trends_interactive.html` (within dashboard): An interactive line plot showing `Global_Sales` trends over `Year_of_Release`.
    
    *   **Insight:** Visualizes the overall evolution of the video game market over time, highlighting periods of growth, decline, or shifts in platform/genre dominance.
        
*   `critic_vs_user_score_interactive.html` (within dashboard): An interactive scatter plot of `Critic_Score` vs `User_Score`, colored by Genre.
    
    *   **Insight:** Helps in understanding "What is the relationship between game ratings (critic and user) and sales performance?" and "Are there discrepancies between critic and user perceptions, and how do these affect sales?" You can identify games where critics and users agree/disagree, and if certain genres have a consistent bias in scores. The hover data (including `Global_Sales`) allows you to see if high scores always translate to high sales, or if there are exceptions.
        

## Future Enhancements

This project provides a robust foundation for video game sales analysis. Here are several logical next steps to make it more robust, scalable, and valuable in a real-world context:

*   **Advanced Feature Engineering:**
    
    *   Extract features from `Name` (e.g., length, presence of keywords).
        
    *   Analyze `Publisher` and `Developer` more deeply (e.g., historical success rates, number of games released).
        
    *   Create interaction features (e.g., `Critic_Score` \* `User_Score`).
        
*   **Time-Series Forecasting:**
    
    *   Given the `Year_of_Release` column, implement time-series models (e.g., ARIMA, Prophet) to forecast future sales trends or predict sales for upcoming years.
        
*   **Model Optimization & Comparison:**
    
    *   Implement hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV) for the Random Forest Regressor.
        
    *   Experiment with other regression algorithms (e.g., Gradient Boosting Machines like XGBoost or LightGBM, Linear Regression, Support Vector Regression) and compare their performance.
        
    *   Implement cross-validation for more robust model evaluation.
        
*   **MLOps Pipeline Integration:**
    
    *   Implement an MLOps (Machine Learning Operations) pipeline using tools like MLflow, DVC, or Kubeflow. This would automate model training, versioning, deployment, and monitoring, ensuring the model remains accurate and up-to-date.
        
*   **Deployment as a Web Application:**
    
    *   Develop a simple web application (using Flask, FastAPI, or Streamlit) where users can input game details and get a real-time sales prediction.
        
*   **Market Segmentation:**
    
    *   Use clustering techniques to identify distinct segments of games or publishers based on their sales patterns, genre focus, or rating profiles.
        

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
    
2.  Create a new branch (`git checkout -b feature/YourFeatureName` or `bugfix/YourBugFix`).
    
3.  Make your changes and ensure tests pass (if applicable).
    
4.  Commit your changes (`git commit -m 'Add new feature X'`).
    
5.  Push to your branch (`git push origin feature/YourFeatureName`).
    
6.  Open a Pull Request, describing your changes in detail.
    

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.