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
    sns.barplot(x=regional_sales.index, y=regional_sales.values, palette='viridis', hue=regional_sales.index, legend=False)
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
    sns.barplot(x=top_platforms.index, y=top_platforms.values, palette='magma', hue=top_platforms.index, legend=False)
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
    sns.barplot(x=top_genres.index, y=top_genres.values, palette='cubehelix', hue=top_genres.index, legend=False)
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
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Saved: correlation_heatmap.png")

    # ADDITION v8: Publisher and Developer Impact Analysis Static Plots
    print("\n--- Generating Static Plots for Publisher/Developer Impact ---")
    publisher_dev_plots_dir = os.path.join(plots_dir, 'publishers_developers')
    os.makedirs(publisher_dev_plots_dir, exist_ok=True)

    # Top 10 Publishers by Total Global Sales
    top_publishers_sales = df.groupby('Publisher')['Global_Sales'].sum().nlargest(10).sort_values(ascending=False)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=top_publishers_sales.index, y=top_publishers_sales.values, palette='viridis', hue=top_publishers_sales.index, legend=False)
    plt.title('Top 10 Publishers by Total Global Sales')
    plt.xlabel('Publisher')
    plt.ylabel('Total Global Sales (Millions)')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(publisher_dev_plots_dir, 'top_10_publishers_sales.png'))
    plt.close()
    print("Saved: publishers_developers/top_10_publishers_sales.png")

    # Top 10 Developers by Total Global Sales (requires 'Developer' column to be clean)
    # Note: Developer column has many missing values, consider imputation or dropping for this analysis
    df_dev = df.dropna(subset=['Developer'])
    if not df_dev.empty:
        top_developers_sales = df_dev.groupby('Developer')['Global_Sales'].sum().nlargest(10).sort_values(ascending=False)
        plt.figure(figsize=(14, 8))
        sns.barplot(x=top_developers_sales.index, y=top_developers_sales.values, palette='cividis', hue=top_developers_sales.index, legend=False)
        plt.title('Top 10 Developers by Total Global Sales')
        plt.xlabel('Developer')
        plt.ylabel('Total Global Sales (Millions)')
        plt.xticks(rotation=60, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(publisher_dev_plots_dir, 'top_10_developers_sales.png'))
        plt.close()
        print("Saved: publishers_developers/top_10_developers_sales.png")
    else:
        print("Skipping Top 10 Developers plot: 'Developer' column has too many missing values after dropping NaNs.")

    # Distribution of Game Counts per Publisher (Top 20)
    top_publishers_count = df['Publisher'].value_counts().nlargest(20)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=top_publishers_count.index, y=top_publishers_count.values, palette='rocket', hue=top_publishers_count.index, legend=False)
    plt.title('Top 20 Publishers by Number of Games Released')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(publisher_dev_plots_dir, 'top_20_publishers_game_count.png'))
    plt.close()
    print("Saved: publishers_developers/top_20_publishers_game_count.png")

    # Average Sales per Game by Publisher (Top 20)
    avg_sales_per_publisher = df.groupby('Publisher')['Global_Sales'].mean().nlargest(20).sort_values(ascending=False)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=avg_sales_per_publisher.index, y=avg_sales_per_publisher.values, palette='mako', hue=avg_sales_per_publisher.index, legend=False)
    plt.title('Top 20 Publishers by Average Global Sales per Game')
    plt.xlabel('Publisher')
    plt.ylabel('Average Global Sales (Millions)')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(publisher_dev_plots_dir, 'top_20_publishers_avg_sales.png'))
    plt.close()
    print("Saved: publishers_developers/top_20_publishers_avg_sales.png")


    # ADDITION v9: Rating (ESRB/PEGI) Analysis Static Plots
    print("\n--- Generating Static Plots for Rating Analysis ---")
    rating_plots_dir = os.path.join(plots_dir, 'ratings')
    os.makedirs(rating_plots_dir, exist_ok=True)

    # Global Sales Distribution by Rating
    # Drop NaNs in 'Rating' for this analysis
    df_rating = df.dropna(subset=['Rating'])
    if not df_rating.empty:
        sales_by_rating = df_rating.groupby('Rating')['Global_Sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sales_by_rating.index, y=sales_by_rating.values, palette='Spectral', hue=sales_by_rating.index, legend=False)
        plt.title('Total Global Sales by Game Rating')
        plt.xlabel('Rating')
        plt.ylabel('Total Global Sales (Millions)')
        plt.tight_layout()
        plt.savefig(os.path.join(rating_plots_dir, 'global_sales_by_rating.png'))
        plt.close()
        print("Saved: ratings/global_sales_by_rating.png")

        # Rating Distribution per Genre (Top Genres)
        # Select top N genres for readability
        top_genres_list = df_rating['Genre'].value_counts().nlargest(10).index.tolist()
        df_top_genres_ratings = df_rating[df_rating['Genre'].isin(top_genres_list)]
        
        if not df_top_genres_ratings.empty:
            # Create a cross-tabulation of Genre and Rating
            genre_rating_counts = pd.crosstab(df_top_genres_ratings['Genre'], df_top_genres_ratings['Rating'], normalize='index')
            genre_rating_counts = genre_rating_counts.loc[top_genres_list] # Ensure consistent order

            plt.figure(figsize=(14, 8))
            genre_rating_counts.plot(kind='bar', stacked=True, cmap='tab20', ax=plt.gca())
            plt.title('Rating Distribution within Top 10 Genres (Proportion)')
            plt.xlabel('Genre')
            plt.ylabel('Proportion of Games')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(rating_plots_dir, 'rating_distribution_per_genre.png'))
            plt.close()
            print("Saved: ratings/rating_distribution_per_genre.png")
        else:
            print("Skipping Rating Distribution per Genre plot: No data for top genres with ratings.")
    else:
        print("Skipping Rating Analysis plots: 'Rating' column has too many missing values after dropping NaNs.")


    # ADDITION v10: Release Quarter/Month Analysis Static Plots
    print("\n--- Generating Static Plots for Release Seasonality ---")
    seasonality_plots_dir = os.path.join(plots_dir, 'seasonality')
    os.makedirs(seasonality_plots_dir, exist_ok=True)

    # Feature Engineering: Extract Month and Quarter from Year_of_Release
    # This dataset doesn't have a specific release date, so we'll assume sales
    # are distributed evenly across the year for simplicity or focus on Year_of_Release trends.
    # If a 'Release_Date' column existed, we would use that.
    # For now, we'll focus on year-level trends which are already covered,
    # or if we want to simulate seasonality, we'd need to add dummy data or assumptions.
    # Given the dataset, direct month/quarter analysis is not feasible without 'Release_Date'.
    # Instead, we can look at the average sales per game per year to see if there are
    # trends in game quality/success over time, or focus on the overall yearly sales trend.

    # Re-using the existing global_sales_trend for interactive plot, as it's the most direct
    # way to show yearly trends. For true seasonality, a 'Release_Date' column is needed.

    # For a static view of yearly sales (already covered by year_of_release_distribution.png,
    # but can be re-emphasized with total sales per year)
    yearly_global_sales = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Year_of_Release', y='Global_Sales', data=yearly_global_sales, marker='o')
    plt.title('Total Global Sales by Year of Release')
    plt.xlabel('Year of Release')
    plt.ylabel('Total Global Sales (Millions)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(seasonality_plots_dir, 'yearly_global_sales_trend.png'))
    plt.close()
    print("Saved: seasonality/yearly_global_sales_trend.png")

    # If we had a 'Release_Date' column, we would do something like:
    # df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
    # df.dropna(subset=['Release_Date'], inplace=True)
    # df['Release_Month'] = df['Release_Date'].dt.month
    # df['Release_Quarter'] = df['Release_Date'].dt.quarter
    # sales_by_month = df.groupby('Release_Month')['Global_Sales'].sum()
    # sales_by_quarter = df.groupby('Release_Quarter')['Global_Sales'].sum()


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

    # 5. Critic vs User Score (Interactive Scatter Plot) with Scaled User Score and Agreement Line
    # ADDITION v5: Scale User_Score for better comparison
    # Ensure 'User_Score' is numeric before scaling. This should be handled in data_cleaning.py,
    # but a defensive check here is good practice.
    if 'User_Score' in df.columns and pd.api.types.is_numeric_dtype(df['User_Score']):
        df['User_Score_Scaled'] = df['User_Score'] * 10
        print("Engineered 'User_Score_Scaled' for visualization.")
    else:
        df['User_Score_Scaled'] = np.nan # Or handle error appropriately
        print("Warning: 'User_Score' is not numeric, cannot create 'User_Score_Scaled'.")

    score_df = df.dropna(subset=['Critic_Score', 'User_Score_Scaled', 'Global_Sales'])

    fig_scores = px.scatter(
        score_df,
        x='Critic_Score',
        y='User_Score_Scaled', # Use the scaled score
        color='Genre',
        size='Global_Sales',
        hover_name='Name',
        hover_data={'Platform': True, 'Publisher': True, 'Year_of_Release': True,
                    'Critic_Score': ':.1f', 'User_Score': ':.1f', # Keep original User_Score in hover
                    'Global_Sales': ':.2f', 'User_Score_Scaled': ':.1f'},
        title='Critic Score vs User Score (Scaled for Comparison, Sized by Global Sales)',
        labels={'Critic_Score': 'Critic Score (out of 100)', 'User_Score_Scaled': 'User Score (Scaled to 100)'}
    )

    # Add a line of perfect agreement (y=x)
    # Ensure the min/max for the line are based on the actual data range for Critic_Score
    min_score = score_df['Critic_Score'].min()
    max_score = score_df['Critic_Score'].max()
    fig_scores.add_trace(go.Scatter(
        x=[min_score, max_score],
        y=[min_score, max_score],
        mode='lines',
        name='Perfect Agreement',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig_scores.write_html(os.path.join(plots_dir, 'critic_vs_user_score_interactive.html'))
    print("Saved: critic_vs_user_score_interactive.html")

    # ADDITION v11: Publisher and Developer Impact Analysis Interactive Plots
    print("\n--- Generating Interactive Plots for Publisher/Developer Impact ---")
    publisher_dev_plots_dir_html = os.path.join(plots_dir, 'publishers_developers')
    os.makedirs(publisher_dev_plots_dir_html, exist_ok=True)

    # Publisher Sales Trends Over Time (Top N Publishers)
    # Select top 10 publishers by global sales for this interactive plot
    top_publishers = df.groupby('Publisher')['Global_Sales'].sum().nlargest(10).index.tolist()
    df_top_publishers = df[df['Publisher'].isin(top_publishers)]
    publisher_yearly_sales = df_top_publishers.groupby(['Year_of_Release', 'Publisher'])['Global_Sales'].sum().reset_index()

    fig_publisher_trends = px.line(
        publisher_yearly_sales,
        x='Year_of_Release',
        y='Global_Sales',
        color='Publisher',
        title='Global Sales Trends for Top 10 Publishers Over Time',
        labels={'Global_Sales': 'Total Global Sales (Millions)'},
        hover_data={'Year_of_Release': True, 'Global_Sales': ':.2f', 'Publisher': True}
    )
    fig_publisher_trends.update_layout(xaxis_title="Year of Release", yaxis_title="Total Global Sales (Millions)")
    fig_publisher_trends.write_html(os.path.join(publisher_dev_plots_dir_html, 'publisher_sales_trends_interactive.html'))
    print("Saved: publishers_developers/publisher_sales_trends_interactive.html")

    # ADDITION v12: Rating (ESRB/PEGI) Analysis Interactive Plots
    print("\n--- Generating Interactive Plots for Rating Analysis ---")
    rating_plots_dir_html = os.path.join(plots_dir, 'ratings')
    os.makedirs(rating_plots_dir_html, exist_ok=True)

    # Sales by Rating per Region (Interactive Grouped Bar Chart)
    df_rating_sales = df.dropna(subset=['Rating'])
    if not df_rating_sales.empty:
        regional_rating_sales = df_rating_sales.groupby('Rating')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()
        fig_regional_rating = px.bar(
            regional_rating_sales,
            x='Rating',
            y=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
            title='Regional Sales by Game Rating',
            labels={'value': 'Total Sales (Millions)', 'variable': 'Region'},
            barmode='group',
            hover_data={'Rating': True, 'value': ':.2f'}
        )
        fig_regional_rating.update_layout(xaxis_tickangle=-45)
        fig_regional_rating.write_html(os.path.join(rating_plots_dir_html, 'regional_sales_by_rating_interactive.html'))
        print("Saved: ratings/regional_sales_by_rating_interactive.html")
    else:
        print("Skipping interactive Regional Sales by Rating plot: 'Rating' column has too many missing values after dropping NaNs.")


    # ADDITION v13: Release Quarter/Month Analysis Interactive Plots
    print("\n--- Generating Interactive Plots for Release Seasonality ---")
    seasonality_plots_dir_html = os.path.join(plots_dir, 'seasonality')
    os.makedirs(seasonality_plots_dir_html, exist_ok=True)

    # For interactive yearly global sales trend (already covered by fig_global_trend, but can be a dedicated plot)
    # This is essentially a re-emphasis of the existing global sales trend, as direct month/quarter
    # analysis requires a 'Release_Date' column not present in the dataset.
    yearly_global_sales_interactive = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
    fig_yearly_sales_interactive = px.line(
        yearly_global_sales_interactive,
        x='Year_of_Release',
        y='Global_Sales',
        title='Total Global Sales Trend by Year (Interactive)',
        labels={'Global_Sales': 'Total Global Sales (Millions)'},
        hover_data={'Year_of_Release': True, 'Global_Sales': ':.2f'}
    )
    fig_yearly_sales_interactive.write_html(os.path.join(seasonality_plots_dir_html, 'yearly_global_sales_trend_interactive.html'))
    print("Saved: seasonality/yearly_global_sales_trend_interactive.html")


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
        'critic_vs_user_score': 'critic_vs_user_score_interactive.html',
        # ADDITION v14: New interactive plots for dashboard
        'publisher_sales_trends': 'publishers_developers/publisher_sales_trends_interactive.html',
        'regional_sales_by_rating': 'ratings/regional_sales_by_rating_interactive.html',
        'yearly_global_sales_trend_interactive': 'seasonality/yearly_global_sales_trend_interactive.html'
    }

    # Read content of each HTML plot
    plot_contents = {}
    for key, filename in plot_files.items():
        file_path = os.path.join(plots_dir, filename)
        # Check for existence of the file before trying to read it
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                plot_contents[key] = f.read()
        else:
            plot_contents[key] = f"<p>Error: {filename} not found. Please ensure EDA script ran successfully and generated this plot.</p>"
            print(f"Warning: {filename} not found for dashboard. Please ensure EDA script ran successfully and generated this plot.")


    # HTML structure for the dashboard
    # The dashboard HTML is updated to include new sections for the new plots.
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
                    <h2 class="card-title">Critic Score vs User Score (Scaled for Comparison, Sized by Global Sales)</h2>
                    {plot_contents['critic_vs_user_score']}
                </div>

                <!-- ADDITION v15: New Sections for Publisher/Developer, Rating, and Seasonality -->
                <div class="card col-span-2">
                    <h2 class="card-title">Publisher Sales Trends Over Time</h2>
                    {plot_contents['publisher_sales_trends']}
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Regional Sales by Game Rating</h2>
                    {plot_contents['regional_sales_by_rating']}
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Total Global Sales Trend by Year (Interactive)</h2>
                    {plot_contents['yearly_global_sales_trend_interactive']}
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

    # Ensure necessary subdirectories for new plots exist
    os.makedirs(os.path.join(STATIC_PLOTS_DIR, 'publishers_developers'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_PLOTS_DIR, 'ratings'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_PLOTS_DIR, 'seasonality'), exist_ok=True)
    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'publishers_developers'), exist_ok=True)
    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'ratings'), exist_ok=True)
    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'seasonality'), exist_ok=True)


    df = load_processed_data(PROCESSED_DATA_PATH)
    if df is not None:
        generate_summary_statistics(df)
        create_static_plots(df, STATIC_PLOTS_DIR)
        create_interactive_plots(df, HTML_PLOTS_DIR)
        create_dashboard(HTML_PLOTS_DIR)
