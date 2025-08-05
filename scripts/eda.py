import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
        print(f"Successfully loaded processed data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def generate_summary_statistics(df):
    """
    Generates and prints descriptive statistics for the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame for which to generate statistics.
    """
    if df is None:
        return

    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all'))
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Data Types ---")
    print(df.info())

def create_static_plots(df, plots_dir='plots/static'):
    """
    Generates static visualizations and saves them as PNG files.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        plots_dir (str): Directory to save static plots.
    """
    if df is None:
        return

    print(f"\n--- Generating Static Plots in {plots_dir} ---")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Global Sales Distribution (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Global_Sales'], bins=50, kde=True)
    plt.title('Distribution of Global Sales')
    plt.xlabel('Global Sales (Millions)')
    plt.ylabel('Number of Games')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'global_sales_distribution.png'))
    plt.close()
    print("Saved: global_sales_distribution.png")

    # 2. Year of Release Distribution (Histogram)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Year_of_Release'], bins=range(int(df['Year_of_Release'].min()), int(df['Year_of_Release'].max()) + 2), kde=False)
    plt.title('Distribution of Games by Year of Release')
    plt.xlabel('Year of Release')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'year_of_release_distribution.png'))
    plt.close()
    print("Saved: year_of_release_distribution.png")

    # 3. Regional Sales Distribution (Bar Plots)
    regional_sales = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=regional_sales.index, y=regional_sales.values, palette='viridis')
    plt.title('Total Regional Sales Distribution')
    plt.xlabel('Region')
    plt.ylabel('Total Sales (Millions)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'regional_sales_distribution.png'))
    plt.close()
    print("Saved: regional_sales_distribution.png")

    # 4. Sales by Top Platforms (Bar Plot)
    top_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(10).sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_platforms.index, y=top_platforms.values, palette='magma')
    plt.title('Top 10 Platforms by Global Sales')
    plt.xlabel('Platform')
    plt.ylabel('Total Global Sales (Millions)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sales_by_platform_static.png'))
    plt.close()
    print("Saved: sales_by_platform_static.png")

    # 5. Sales by Top Genres (Bar Plot)
    top_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(10).sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_genres.index, y=top_genres.values, palette='cubehelix')
    plt.title('Top 10 Genres by Global Sales')
    plt.xlabel('Genre')
    plt.ylabel('Total Global Sales (Millions)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_genres_static.png'))
    plt.close()
    print("Saved: top_genres_static.png")

    # 6. Correlation Heatmap
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude 'Year_of_Release' from correlation heatmap if it's treated as a categorical-like feature
    # or if its linear correlation with sales is not the primary interest.
    # For this project, we'll include it as it's a numeric column.
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Saved: correlation_heatmap.png")


def create_interactive_plots(df, plots_dir='plots/html'):
    """
    Generates interactive Plotly visualizations and saves them as HTML files.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        plots_dir (str): Directory to save interactive plots.
    """
    if df is None:
        return

    print(f"\n--- Generating Interactive Plots in {plots_dir} ---")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Regional Sales by Genre (Interactive Grouped Bar Chart)
    # Aggregate sales by Genre and Region
    regional_genre_sales = df.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()
    fig_regional_genre = px.bar(
        regional_genre_sales,
        x='Genre',
        y=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
        title='Regional Sales by Genre',
        labels={'value': 'Total Sales (Millions)', 'variable': 'Region'},
        barmode='group',
        hover_data={'Genre': True, 'value': ':.2f'}
    )
    fig_regional_genre.update_layout(xaxis_tickangle=-45)
    fig_regional_genre.write_html(os.path.join(plots_dir, 'regional_sales_by_genre_interactive.html'))
    print("Saved: regional_sales_by_genre_interactive.html")

    # 2. Regional Sales Trends Over Time (Interactive Line Plot)
    # Aggregate sales by Year_of_Release and Region
    regional_year_sales = df.groupby('Year_of_Release')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum().reset_index()
    fig_regional_trends = px.line(
        regional_year_sales,
        x='Year_of_Release',
        y=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'],
        title='Regional and Global Sales Trends Over Time',
        labels={'value': 'Total Sales (Millions)', 'variable': 'Region/Global'},
        hover_data={'Year_of_Release': True, 'value': ':.2f'}
    )
    fig_regional_trends.update_layout(xaxis_title="Year of Release", yaxis_title="Total Sales (Millions)")
    fig_regional_trends.write_html(os.path.join(plots_dir, 'regional_sales_trends_interactive.html'))
    print("Saved: regional_sales_trends_interactive.html")

    # 3. Sales by Platform (Interactive Bar Chart)
    platform_sales = df.groupby('Platform')['Global_Sales'].sum().reset_index().sort_values(by='Global_Sales', ascending=False)
    fig_platform_sales = px.bar(
        platform_sales,
        x='Platform',
        y='Global_Sales',
        title='Total Global Sales by Platform',
        labels={'Global_Sales': 'Total Global Sales (Millions)'},
        hover_data={'Platform': True, 'Global_Sales': ':.2f'}
    )
    fig_platform_sales.update_layout(xaxis_tickangle=-45)
    fig_platform_sales.write_html(os.path.join(plots_dir, 'sales_by_platform_interactive.html'))
    print("Saved: sales_by_platform_interactive.html")

    # 4. Global Sales Trends Over Time (Interactive Line Plot)
    global_sales_trend = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
    fig_global_trend = px.line(
        global_sales_trend,
        x='Year_of_Release',
        y='Global_Sales',
        title='Global Sales Trends Over Time',
        labels={'Global_Sales': 'Total Global Sales (Millions)'},
        hover_data={'Year_of_Release': True, 'Global_Sales': ':.2f'}
    )
    fig_global_trend.update_layout(xaxis_title="Year of Release", yaxis_title="Total Global Sales (Millions)")
    fig_global_trend.write_html(os.path.join(plots_dir, 'sales_trends_interactive.html'))
    print("Saved: sales_trends_interactive.html")

    # 5. Critic vs User Score (Interactive Scatter Plot)
    # Filter out rows where scores might be missing or zero after cleaning, if any.
    score_df = df.dropna(subset=['Critic_Score', 'User_Score', 'Global_Sales'])
    fig_scores = px.scatter(
        score_df,
        x='Critic_Score',
        y='User_Score',
        color='Genre',
        size='Global_Sales', # Size markers by global sales
        hover_name='Name',
        hover_data={'Platform': True, 'Publisher': True, 'Year_of_Release': True,
                    'Critic_Score': ':.1f', 'User_Score': ':.1f', 'Global_Sales': ':.2f'},
        title='Critic Score vs User Score (Sized by Global Sales)',
        labels={'Critic_Score': 'Critic Score (out of 100)', 'User_Score': 'User Score (out of 10)'}
    )
    fig_scores.write_html(os.path.join(plots_dir, 'critic_vs_user_score_interactive.html'))
    print("Saved: critic_vs_user_score_interactive.html")


def create_dashboard(plots_dir='plots/html'):
    """
    Creates a single HTML dashboard file that embeds the interactive Plotly graphs.
    This dashboard provides a consolidated view of key insights.

    Args:
        plots_dir (str): Directory where interactive plots are saved and dashboard will be saved.
    """
    print(f"\n--- Creating Interactive Dashboard in {plots_dir} ---")
    dashboard_path = os.path.join(plots_dir, 'dashboard.html')

    # Paths to individual interactive plots relative to the dashboard.html
    plot_files = {
        'regional_sales_by_genre': 'regional_sales_by_genre_interactive.html',
        'regional_sales_trends': 'regional_sales_trends_interactive.html',
        'sales_by_platform': 'sales_by_platform_interactive.html',
        'sales_trends': 'sales_trends_interactive.html',
        'critic_vs_user_score': 'critic_vs_user_score_interactive.html'
    }

    # Read content of each HTML plot
    plot_contents = {}
    for key, filename in plot_files.items():
        file_path = os.path.join(plots_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                plot_contents[key] = f.read()
        else:
            plot_contents[key] = f"<p>Error: {filename} not found. Please ensure EDA script ran successfully.</p>"
            print(f"Warning: {filename} not found for dashboard.")

    # HTML structure for the dashboard
    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Game Sales Analysis Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f3f4f6;
                color: #374151;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }}
            .card {{
                background-color: #ffffff;
                border-radius: 0.75rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 1.5rem;
                margin-bottom: 2rem;
            }}
            .card-title {{
                font-size: 1.5rem;
                font-weight: 600;
                color: #1f2937;
                margin-bottom: 1rem;
                border-bottom: 2px solid #e5e7eb;
                padding-bottom: 0.5rem;
            }}
            /* Adjust Plotly plots to be responsive */
            .plotly-graph-div {{
                width: 100% !important;
                height: auto !important;
                min-height: 400px; /* Ensure a minimum height for visibility */
            }}
        </style>
    </head>
    <body class="p-4">
        <div class="container">
            <h1 class="text-4xl font-bold text-center text-gray-800 mb-8 rounded-lg bg-white p-4 shadow-md">
                Video Game Sales Interactive Dashboard
            </h1>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="card col-span-2">
                    <h2 class="card-title">Regional Sales by Genre</h2>
                    {plot_contents['regional_sales_by_genre']}
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Regional and Global Sales Trends Over Time</h2>
                    {plot_contents['regional_sales_trends']}
                </div>

                <div class="card">
                    <h2 class="card-title">Total Global Sales by Platform</h2>
                    {plot_contents['sales_by_platform']}
                </div>

                <div class="card">
                    <h2 class="card-title">Global Sales Trends Over Time</h2>
                    {plot_contents['sales_trends']}
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Critic Score vs User Score (Sized by Global Sales)</h2>
                    {plot_contents['critic_vs_user_score']}
                </div>
            </div>

            <footer class="text-center text-gray-500 mt-8 text-sm">
                <p>&copy; 2023 Video Game Sales Analysis. All rights reserved.</p>
            </footer>
        </div>
    </body>
    </html>
    """

    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"Interactive dashboard saved to {dashboard_path}")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_data.csv')
    STATIC_PLOTS_DIR = os.path.join('plots', 'static')
    HTML_PLOTS_DIR = os.path.join('plots', 'html')

    df = load_processed_data(PROCESSED_DATA_PATH)
    if df is not None:
        generate_summary_statistics(df)
        create_static_plots(df, STATIC_PLOTS_DIR)
        create_interactive_plots(df, HTML_PLOTS_DIR)
        create_dashboard(HTML_PLOTS_DIR)
