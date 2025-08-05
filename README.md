# Video Game Sales Analysis & Prediction


## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Technologies Used](#technologies-used)
4.  [File and Directory Structure](#file-and-directory-structure)
5.  [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Download the Dataset](#download-the-dataset)
    * [Usage](#usage)
6.  [Exploratory Data Analysis (EDA) Insights](#exploratory-data-analysis-eda-insights)
    * [Overall Sales and Release Trends](#overall-sales-and-release-trends)
    * [Platform and Genre Performance](#platform-and-genre-performance)
    * [Critic vs. User Scores](#critic-vs-user-scores)
    * [Publisher and Developer Impact](#publisher-and-developer-impact)
    * [Game Rating Analysis](#game-rating-analysis)
7.  [Predictive Model](#predictive-model)
8.  [Future Enhancements](#future-enhancements)
9.  [Contributing](#contributing)
10. [License](#license)


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
        
*   **Interactive Visualizations:** Production of dynamic, interactive plots using Plotly, saved as standalone JSON files, allowing for deeper data exploration (zooming, panning, hovering for details on sales trends, genre performance, and score comparisons).
    
*   **Interactive Mini-Dashboard:** A single HTML file (`dashboard.html`) that aggregates key interactive plots by dynamically loading their JSON data, offering a concise and engaging overview of the dataset's insights into industry trends and game performance.
    
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

# 

    .
    ├── data/
    │   ├── raw/
    │   │   └── video_game_sales_and_ratings.csv  # Raw dataset (download separately)
    │   └── processed/
    │       └── processed_data.csv   # Cleaned and featurized data
    ├── plots/
    │   ├── html/
    │   │   ├── best_games/
    │   │   │   ├── top_25_critic_score_sales_interactive.html
    │   │   │   ├── top_25_global_sales_and_scores_interactive.html
    │   │   │   └── top_25_global_sales_and_scores_interactive.json
    │   │   ├── genre_metrics/
    │   │   │   ├── critic_score_by_genre_boxplot_interactive.html
    │   │   │   ├── critic_score_by_genre_boxplot_interactive.json
    │   │   │   ├── global_sales_by_genre_boxplot_interactive.json
    │   │   │   ├── global_sales_by_genre_boxplot_interactive_log.html
    │   │   │   ├── user_score_by_genre_boxplot_interactive.html
    │   │   │   └── user_score_by_genre_boxplot_interactive.json
    │   │   ├── publishers_developers/
    │   │   │   ├── avg_sales_per_publisher_interactive.html
    │   │   │   ├── avg_sales_per_publisher_interactive.json
    │   │   │   ├── publisher_sales_trends_interactive.html
    │   │   │   ├── publisher_sales_trends_interactive.json
    │   │   │   ├── top_25_publishers_sales_scores_interactive.html
    │   │   │   └── top_25_publishers_sales_scores_interactive.json
    │   │   ├── ratings/
    │   │   │   ├── rating_distribution_per_genre_interactive.html
    │   │   │   ├── rating_distribution_per_genre_interactive.json
    │   │   │   ├── regional_sales_by_rating_interactive.html
    │   │   │   └── regional_sales_by_rating_interactive.json
    │   │   ├── seasonality/
    │   │   │   ├── yearly_global_sales_trend_interactive.html
    │   │   │   └── yearly_global_sales_trend_interactive.json
    │   │   ├── critic_vs_user_score_density_interactive.html
    │   │   ├── critic_vs_user_score_interactive.html
    │   │   ├── critic_vs_user_score_scatter_enhanced_interactive.html
    │   │   ├── critic_vs_user_score_scatter_enhanced_interactive.json
    │   │   ├── dashboard.html  # Interactive mini-dashboard
    │   │   ├── regional_sales_by_genre_interactive.html
    │   │   ├── regional_sales_by_genre_interactive.json
    │   │   ├── regional_sales_trends_interactive.html
    │   │   ├── regional_sales_trends_interactive.json
    │   │   ├── sales_by_platform_interactive.html
    │   │   ├── sales_by_platform_interactive.json
    │   │   ├── sales_trends_interactive.html
    │   │   └── sales_trends_interactive.json
    │   └── static/
    │       ├── genre_metrics/
    │       │   ├── critic_score_by_genre_boxplot.png
    │       │   └── global_sales_by_genre_boxplot_log.png
    │       ├── publishers_developers/
    │       │   ├── top_10_developers_sales.png
    │       │   ├── top_10_publishers_sales.png
    │       │   ├── top_20_publishers_avg_sales.png
    │       │   └── top_20_publishers_game_count.png
    │       ├── ratings/
    │       │   ├── global_sales_by_rating.png
    │       │   └── rating_distribution_per_genre.png
    │       ├── seasonality/
    │       │   └── yearly_global_sales_trend.png
    │       ├── best_games/
    │       │   └── top_25_critic_score_sales_static.png
    │       ├── correlation_heatmap.png
    │       ├── global_sales_distribution.png
    │       ├── regional_sales_distribution.png
    │       ├── sales_by_platform_static.png
    │       ├── top_genres_static.png
    │       └── year_of_release_distribution.png
    ├── scripts/
    │   ├── data_cleaning.py
    │   ├── eda.py
    │   ├── predict_sales.py
    │   └── train_model.py
    ├── models/
    │   ├── game_sales_model.pkl    # Trained machine learning model
    │   └── label_encoders.pkl      # Saved label encoders for categorical features
    ├── .gitignore
    ├── main.py                     # Orchestrates the entire workflow
    └── requirements.txt
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
        
    

## Exploratory Data Analysis (EDA) Insights

### Overall Sales and Release Trends

*   **Global Sales Distribution**: The distribution of global sales is highly skewed, with a large number of games having low sales and a long tail of a few blockbuster titles with very high sales. This highlights the "hit-driven" nature of the video game industry.
    
*   **Year of Release Distribution**: The volume of game releases shows a clear peak around the mid-2000s to early 2010s, likely corresponding to the peak of the 7th generation of consoles (PS3, Xbox 360, Wii). There's a noticeable decline in recent years, which might be due to data limitations or shifts in market dynamics (e.g., digital-only releases not fully captured).
    
*   **Regional Sales Distribution**: North America (NA\_Sales) and Europe (EU\_Sales) consistently represent the largest markets for video games, followed by Japan (JP\_Sales) and other regions. This regional disparity is crucial for targeted marketing and localization strategies.
    
*   **Sales Trends Over Time**: Global sales generally peaked around 2008-2009, aligning with the console generation peak, and have shown a gradual decline or stabilization since then in the dataset, reflecting the evolving market.
    

### Platform and Genre Performance

*   **Top Platforms by Global Sales**: Platforms like PS2, X360, PS3, Wii, and DS have historically generated the highest global sales, indicating their immense popularity during their active lifespans. Newer generations are also emerging strongly.
    
*   **Top Genres by Global Sales**: Action, Sports, Shooter, and Role-Playing are consistently the highest-grossing genres globally, suggesting strong and sustained consumer demand in these categories.
    
*   **Regional Sales by Genre**: Genre preferences vary significantly by region. For example, Role-Playing games hold a notably larger share in Japan compared to other regions, underscoring the unique gaming tastes in Japan, with a strong preference for RPGs. In North America and Europe, focusing on popular genres like Shooter, Sports, and Action could be more successful.
    

### Critic vs. User Scores

*   **Correlation with Sales**: The relationship between critic/user scores and global sales is positive but not always linear or strong across all genres/platforms. While high scores generally correlate with better sales, there are exceptions, indicating that other factors like marketing, brand loyalty, or unique features also play a significant role.
    
*   **Agreement between Scores**: There's a general positive correlation between Critic Scores and User Scores, suggesting that critics and users often agree on game quality. However, discrepancies exist, which can be valuable for understanding different audience perspectives.
    

### Publisher and Developer Impact

*   **Top Publishers by Total Global Sales**: Companies like Electronic Arts, Activision, Nintendo, Sony, and Ubisoft dominate the market in terms of cumulative global sales, reflecting their long-standing presence and successful franchises.
    
*   **Top Developers by Total Global Sales**: Similar to publishers, certain development studios consistently produce high-selling titles, demonstrating their creative and technical prowess.
    
*   **Average Sales per Game by Publisher**: Analyzing average sales per game helps identify publishers who consistently produce high-quality or high-selling individual titles, rather than just high volume.
    

### Game Rating Analysis

*   **Global Sales by Rating**: Games with "E" (Everyone) and "T" (Teen) ratings tend to generate the highest global sales, indicating a broader market appeal for family-friendly or moderately-rated content. "M" (Mature) rated games also contribute significantly, especially in certain genres.
    
*   **Rating Distribution per Genre**: Certain genres predominantly target specific age groups or content levels. For example, Shooter games are often rated "M," while Sports games are typically "E" or "E10+". This insight helps in understanding content trends and target audiences within successful genres.
    

## Predictive Model

A Random Forest Regressor model has been trained to predict `Global_Sales` based on various game attributes. The model provides a reasonable prediction for unseen data, though performance can be further enhanced.

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