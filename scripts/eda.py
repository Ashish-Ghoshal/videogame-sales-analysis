import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import json # Import json module

# Custom JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that extends the default JSONEncoder to handle NumPy arrays.
    It converts NumPy arrays into Python lists, which are JSON serializable.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    Generates and returns key summary statistics as an HTML string,
    tailored for a managerial overview.

    Args:
        df (pd.DataFrame): The DataFrame for which to generate statistics.

    Returns:
        str: An HTML string containing key summary statistics.
    """
    if df is None:
        return "<p>Error: No data available for summary statistics.</p>"

    # --- Key Business Metrics ---
    total_games = len(df)
    total_global_sales = df['Global_Sales'].sum()
    
    # Ensure scores are numeric before calculating means
    df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

    avg_critic_score = df['Critic_Score'].mean()
    avg_user_score = df['User_Score'].mean() * 10 # Scale user score to 100 for consistency if not already
    
    unique_platforms = df['Platform'].nunique()
    unique_genres = df['Genre'].nunique()
    unique_publishers = df['Publisher'].nunique()
    
    earliest_year = int(df['Year_of_Release'].min()) if not pd.isna(df['Year_of_Release'].min()) else 'N/A'
    latest_year = int(df['Year_of_Release'].max()) if not pd.isna(df['Year_of_Release'].max()) else 'N/A'

    # Top 3 Genres by Global Sales
    top_3_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(3)
    top_3_genres_html = ""
    for genre, sales in top_3_genres.items():
        top_3_genres_html += f"<li class='flex justify-between items-center'><span class='text-gray-700 font-medium'>{genre}</span> <span class='text-blue-600 font-semibold'>{sales:.2f}M</span></li>"

    # Top 3 Platforms by Global Sales
    top_3_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(3)
    top_3_platforms_html = ""
    for platform, sales in top_3_platforms.items():
        top_3_platforms_html += f"<li class='flex justify-between items-center'><span class='text-gray-700 font-medium'>{platform}</span> <span class='text-blue-600 font-semibold'>{sales:.2f}M</span></li>"

    summary_html = f"""
    <div class="p-4 bg-white rounded-lg shadow-md">
        <h3 class="text-xl font-semibold text-gray-800 mb-4 border-b pb-2">Key Dataset Metrics</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <div class="bg-indigo-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Total Games Analyzed</p>
                <p class="text-2xl font-bold text-indigo-700">{total_games:,}</p>
            </div>
            <div class="bg-green-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Total Global Sales</p>
                <p class="text-2xl font-bold text-green-700">{total_global_sales:,.2f}M USD</p>
            </div>
            <div class="bg-yellow-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Avg. Critic Score (out of 100)</p>
                <p class="text-2xl font-bold text-yellow-700">{avg_critic_score:.1f}</p>
            </div>
            <div class="bg-red-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Avg. User Score (out of 100)</p>
                <p class="text-2xl font-bold text-red-700">{avg_user_score:.1f}</p>
            </div>
            <div class="bg-purple-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Unique Platforms</p>
                <p class="text-2xl font-bold text-purple-700">{unique_platforms}</p>
            </div>
            <div class="bg-teal-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Unique Genres</p>
                <p class="text-2xl font-bold text-teal-700">{unique_genres}</p>
            </div>
            <div class="bg-orange-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Unique Publishers</p>
                <p class="text-2xl font-bold text-orange-700">{unique_publishers}</p>
            </div>
            <div class="bg-blue-50 p-3 rounded-lg shadow-sm">
                <p class="text-sm text-gray-500">Data Range (Years)</p>
                <p class="text-2xl font-bold text-blue-700">{earliest_year} - {latest_year}</p>
            </div>
        </div>

        <h3 class="text-xl font-semibold text-gray-800 mt-6 mb-4 border-b pb-2">Top 3 Genres by Global Sales</h3>
        <ul class="space-y-2">
            {top_3_genres_html}
        </ul>
        <p class="text-gray-600 text-sm mt-2">
            These are the genres that have generated the most revenue across all regions.
        </p>

        <h3 class="text-xl font-semibold text-gray-800 mt-6 mb-4 border-b pb-2">Top 3 Platforms by Global Sales</h3>
        <ul class="space-y-2">
            {top_3_platforms_html}
        </ul>
        <p class="text-gray-600 text-sm mt-2">
            These platforms represent the highest revenue generators in the video game market.
        </p>
    </div>
    """
    return summary_html


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

    # Business Question: What is the overall distribution of game sales?
    # Insight: Helps understand the common sales figures and identify blockbuster outliers.
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

    # Business Question: How has the volume of game releases changed over the years?
    # Insight: Reveals periods of industry growth or specific console generation peaks.
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

    # Business Question: Which regions are the largest markets for video games?
    # Insight: Provides a quick overview of market dominance by geography.
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

    # Business Question: Which gaming platforms have generated the most revenue?
    # Insight: Identifies leading platforms and their historical market share.
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

    # Business Question: What are the most commercially successful game genres?
    # Insight: Helps in understanding market demand and potential for new game development.
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

    # Business Question: What are the linear relationships between numerical game attributes?
    # Insight: Identifies strong correlations (e.g., between regional sales, or scores and sales)
    # which can inform predictive modeling and strategic focus.
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

    # Publisher and Developer Impact Analysis Static Plots
    print("\n--- Generating Static Plots for Publisher/Developer Impact ---")
    publisher_dev_plots_dir = os.path.join(plots_dir, 'publishers_developers')
    os.makedirs(publisher_dev_plots_dir, exist_ok=True)

    # Business Question: Which publishers have generated the most overall revenue?
    # Insight: Identifies market leaders and their cumulative impact on the industry.
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

    # Business Question: Which developers are responsible for the highest-selling games?
    # Insight: Highlights key development studios and their track record of success.
    # Top 10 Developers by Total Global Sales (requires 'Developer' column to be clean)
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

    # Business Question: Which publishers are most prolific in terms of game releases?
    # Insight: Differentiates between publishers focusing on many releases vs. fewer, high-impact titles.
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

    # Business Question: Which publishers achieve the highest average sales per game?
    # Insight: Identifies publishers with a strong track record of producing successful individual titles.
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


    # Rating (ESRB/PEGI) Analysis Static Plots
    print("\n--- Generating Static Plots for Rating Analysis ---")
    rating_plots_dir = os.path.join(plots_dir, 'ratings')
    os.makedirs(rating_plots_dir, exist_ok=True)

    # Business Question: Which game ratings (e.g., E, T, M) generate the most revenue globally?
    # Insight: Helps understand market demand for content suitability.
    # Global Sales Distribution by Rating
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

        # Business Question: How do game ratings distribute across different genres?
        # Insight: Reveals if certain genres predominantly target specific age groups or content levels.
        # Rating Distribution per Genre (Top Genres)
        top_genres_list = df_rating['Genre'].value_counts().nlargest(10).index.tolist()
        df_top_genres_ratings = df_rating[df_rating['Genre'].isin(top_genres_list)]
        
        if not df_top_genres_ratings.empty:
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


    # Release Quarter/Month Analysis Static Plots
    print("\n--- Generating Static Plots for Release Seasonality ---")
    seasonality_plots_dir = os.path.join(plots_dir, 'seasonality')
    os.makedirs(seasonality_plots_dir, exist_ok=True)

    # Business Question: How has total global sales evolved over the years?
    # Insight: Shows overall industry growth or decline trends.
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

    # Business Question: How do average critic and user scores vary by genre?
    # Insight: Helps understand general quality perception across different game types.
    # Box Plots of Scores/Sales by Genre
    print("\n--- Generating Box Plots for Scores/Sales by Genre ---")
    genre_metrics_plots_dir = os.path.join(plots_dir, 'genre_metrics')
    os.makedirs(genre_metrics_plots_dir, exist_ok=True)

    # Box plot for Critic Score by Genre
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='Genre', y='Critic_Score', data=df, palette='pastel', hue='Genre', legend=False)
    plt.title('Distribution of Critic Scores by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Critic Score (out of 100)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(genre_metrics_plots_dir, 'critic_score_by_genre_boxplot.png'))
    plt.close()
    print("Saved: genre_metrics/critic_score_by_genre_boxplot.png")

    # Box plot for User Score by Genre (using scaled score for consistency)
    if 'User_Score_Scaled' in df.columns:
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='Genre', y='User_Score_Scaled', data=df, palette='pastel', hue='Genre', legend=False)
        plt.title('Distribution of User Scores (Scaled) by Genre')
        plt.xlabel('Genre')
        plt.ylabel('User Score (Scaled to 100)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(genre_metrics_plots_dir, 'user_score_by_genre_boxplot.png'))
        plt.close()
        print("Saved: genre_metrics/user_score_by_genre_boxplot.png")
    else:
        print("Skipping User Score by Genre Boxplot: 'User_Score_Scaled' not available.")

    # Box plot for Global Sales by Genre (using log scale due to skewness)
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='Genre', y='Global_Sales', data=df, palette='pastel', hue='Genre', legend=False)
    plt.yscale('log') # Use log scale for sales due to high skewness
    plt.title('Distribution of Global Sales by Genre (Log Scale)')
    plt.xlabel('Genre')
    plt.ylabel('Global Sales (Millions, Log Scale)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(genre_metrics_plots_dir, 'global_sales_by_genre_boxplot_log.png'))
    plt.close()
    print("Saved: genre_metrics/global_sales_by_genre_boxplot_log.png")


def create_interactive_plots(df, plots_dir='plots/html'):
    """
    Generates interactive Plotly visualizations and saves their JSON data.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        plots_dir (str): Directory to save interactive plot JSON files.
    """
    if df is None:
        return

    print(f"\n--- Generating Interactive Plots in {plots_dir} ---")
    os.makedirs(plots_dir, exist_ok=True)

    # Business Question: How do genre sales vary across different regions?
    # Insight: Identifies regional market preferences for game genres.
    # 1. Regional Sales by Genre (Interactive Grouped Bar Chart)
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
    with open(os.path.join(plots_dir, 'regional_sales_by_genre_interactive.json'), 'w') as f:
        json.dump(fig_regional_genre.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: regional_sales_by_genre_interactive.json")

    # Business Question: How have regional and global sales trends evolved over time?
    # Insight: Shows market growth, decline, and relative importance of regions over years.
    # 2. Regional Sales Trends Over Time (Interactive Line Plot)
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
    with open(os.path.join(plots_dir, 'regional_sales_trends_interactive.json'), 'w') as f:
        json.dump(fig_regional_trends.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: regional_sales_trends_interactive.json")

    # Business Question: Which platforms have generated the most global sales?
    # Insight: Identifies dominant platforms in the overall market.
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
    with open(os.path.join(plots_dir, 'sales_by_platform_interactive.json'), 'w') as f:
        json.dump(fig_platform_sales.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: sales_by_platform_interactive.json")

    # Business Question: What is the overall trend of global video game sales over time?
    # Insight: Shows the industry's historical growth or decline.
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
    fig_global_trend.update_layout(xaxis_title="Year of Release", yaxis_title="Total Sales (Millions)")
    with open(os.path.join(plots_dir, 'sales_trends_interactive.json'), 'w') as f:
        json.dump(fig_global_trend.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: sales_trends_interactive.json")

    # Business Question: How do critic and user scores relate, and how does this impact sales?
    # Insight: Reveals agreement/disagreement between critics and users, and how this correlates with commercial success.
    # 5. Critic vs User Score (Interactive Scatter Plot) with Scaled User Score and Agreement Line
    if 'User_Score' in df.columns and pd.api.types.is_numeric_dtype(df['User_Score']):
        df['User_Score_Scaled'] = df['User_Score'] * 10
        print("Engineered 'User_Score_Scaled' for visualization.")
    else:
        df['User_Score_Scaled'] = np.nan
        print("Warning: 'User_Score' is not numeric, cannot create 'User_Score_Scaled'.")

    score_df = df.dropna(subset=['Critic_Score', 'User_Score_Scaled', 'Global_Sales'])

    fig_scores_scatter_enhanced = px.scatter(
        score_df,
        x='Critic_Score',
        y='User_Score_Scaled', # Use the scaled score
        color='Genre',
        size='Global_Sales',
        hover_name='Name',
        hover_data={'Platform': True, 'Publisher': True, 'Year_of_Release': True,
                    'Critic_Score': ':.1f', 'User_Score': ':.1f', # Keep original User_Score in hover
                    'Global_Sales': ':.2f', 'User_Score_Scaled': ':.1f'},
        title='Critic Score vs User Score (Scaled, Sized by Sales, by Genre)',
        labels={'Critic_Score': 'Critic Score (out of 100)', 'User_Score_Scaled': 'User Score (Scaled to 100)'},
        opacity=0.6, # Make points semi-transparent to see density
    )
    min_score = score_df['Critic_Score'].min()
    max_score = score_df['Critic_Score'].max()
    fig_scores_scatter_enhanced.add_trace(go.Scatter(
        x=[min_score, max_score],
        y=[min_score, max_score],
        mode='lines',
        name='Perfect Agreement',
        line=dict(color='red', dash='dash', width=2)
    ))
    with open(os.path.join(plots_dir, 'critic_vs_user_score_scatter_enhanced_interactive.json'), 'w') as f:
        json.dump(fig_scores_scatter_enhanced.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: critic_vs_user_score_scatter_enhanced_interactive.json")


    # Publisher and Developer Impact Analysis Interactive Plots
    print("\n--- Generating Interactive Plots for Publisher/Developer Impact ---")
    publisher_dev_plots_dir_html = os.path.join(plots_dir, 'publishers_developers')
    os.makedirs(publisher_dev_plots_dir_html, exist_ok=True)

    # Business Question: How do the sales of top publishers trend over time?
    # Insight: Reveals the long-term performance and market presence of major publishers.
    # Publisher Sales Trends Over Time (Top N Publishers)
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
    fig_publisher_trends.update_layout(xaxis_title="Year of Release", yaxis_title="Total Sales (Millions)")
    with open(os.path.join(publisher_dev_plots_dir_html, 'publisher_sales_trends_interactive.json'), 'w') as f:
        json.dump(fig_publisher_trends.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: publishers_developers/publisher_sales_trends_interactive.json")

    # Business Question: What is the average sales performance per game for top publishers?
    # Insight: Helps identify publishers that consistently produce high-quality or high-selling titles.
    # Interactive Bar Chart for Average Sales per Game by Publisher
    avg_sales_per_publisher_interactive = df.groupby('Publisher')['Global_Sales'].mean().nlargest(20).sort_values(ascending=False).reset_index()
    fig_avg_sales_publisher = px.bar(
        avg_sales_per_publisher_interactive,
        x='Publisher',
        y='Global_Sales',
        title='Top 20 Publishers by Average Global Sales per Game',
        labels={'Global_Sales': 'Average Global Sales (Millions)'},
        hover_data={'Publisher': True, 'Global_Sales': ':.2f'}
    )
    fig_avg_sales_publisher.update_layout(xaxis_tickangle=-45)
    with open(os.path.join(publisher_dev_plots_dir_html, 'avg_sales_per_publisher_interactive.json'), 'w') as f:
        json.dump(fig_avg_sales_publisher.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: publishers_developers/avg_sales_per_publisher_interactive.json")

    # NEW PLOT: Top 25 Publishers by Global Sales with Average Critic and User Scores
    print("\n--- Generating Interactive Top 25 Publishers by Global Sales and Scores Plot ---")
    # Business Question: Which publishers are the highest-selling globally, and how do their average critic and user scores compare?
    # Insight: Identifies top commercial publishers and the alignment of average critical/user reception with their overall sales.
    
    # Data preparation: Aggregate by Publisher
    publisher_metrics = df.groupby('Publisher').agg(
        Global_Sales=('Global_Sales', 'sum'),
        Critic_Score=('Critic_Score', 'mean'),
        User_Score=('User_Score', 'mean')
    ).reset_index()

    # Scale Critic_Score to be out of 10
    publisher_metrics['Critic_Score'] = publisher_metrics['Critic_Score'] / 10

    # Drop NaNs that might result from aggregation if a publisher has no scores
    publisher_metrics.dropna(subset=['Critic_Score', 'User_Score', 'Global_Sales'], inplace=True)

    # Get top 25 publishers by Global_Sales
    t25_publishers = publisher_metrics.sort_values('Global_Sales', ascending=False).head(25).reset_index(drop=True)

    # Create a figure with secondary Y-axis
    fig_top_publishers_sales_scores = go.Figure()

    # Add bar trace for Global Sales
    fig_top_publishers_sales_scores.add_trace(go.Bar(
        x=t25_publishers['Publisher'],
        y=t25_publishers['Global_Sales'],
        name='Total Global Sales',
        marker_color='rgba(0,0,0,0)', # Transparent fill
        marker_line_color='blue', # Border color
        marker_line_width=1,
        opacity=0.8,
        hoverinfo='x+y',
        hovertemplate='<b>%{x}</b><br>Total Sales: %{y:.2f}M<extra></extra>'
    ))

    # Add scatter trace for Critic Score on secondary Y-axis
    fig_top_publishers_sales_scores.add_trace(go.Scatter(
        x=t25_publishers['Publisher'],
        y=t25_publishers['Critic_Score'],
        name='Avg Critic Score',
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        yaxis='y2',
        hoverinfo='x+y',
        hovertemplate='<b>%{x}</b><br>Avg Critic Score: %{y:.1f}<extra></extra>'
    ))

    # Add scatter trace for User Score on secondary Y-axis
    fig_top_publishers_sales_scores.add_trace(go.Scatter(
        x=t25_publishers['Publisher'],
        y=t25_publishers['User_Score'],
        name='Avg User Score',
        mode='markers',
        marker=dict(color='green', size=10, symbol='circle'),
        yaxis='y2',
        hoverinfo='x+y',
        hovertemplate='<b>%{x}</b><br>Avg User Score: %{y:.1f}<extra></extra>'
    ))

    # Update layout for dual Y-axes
    fig_top_publishers_sales_scores.update_layout(
        title_text='Top 25 Publishers by Global Sales with Average Critic and User Scores (Interactive)',
        xaxis_title='Publisher Name',
        yaxis=dict(
            title='Total Global Sales (Millions)',
            title_font=dict(color='blue'),
            tickfont=dict(color='blue'),
            side='left'
        ),
        yaxis2=dict(
            title='Average Score (out of 10)',
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            overlaying='y',
            side='right',
            range=[0, 10] # Scores are out of 10
        ),
        hovermode='x unified',
        xaxis_tickangle=-45,
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        height=700 # Increased height for better readability
    )

    with open(os.path.join(publisher_dev_plots_dir_html, 'top_25_publishers_sales_scores_interactive.json'), 'w') as f:
        json.dump(fig_top_publishers_sales_scores.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: publishers_developers/top_25_publishers_sales_scores_interactive.json")


    # Rating (ESRB/PEGI) Analysis Interactive Plots
    print("\n--- Generating Interactive Plots for Rating Analysis ---")
    rating_plots_dir_html = os.path.join(plots_dir, 'ratings')
    os.makedirs(rating_plots_dir_html, exist_ok=True)

    # Business Question: How do sales of games with different ratings vary across regions?
    # Insight: Understands regional market acceptance and demand for specific content ratings.
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
        with open(os.path.join(rating_plots_dir_html, 'regional_sales_by_rating_interactive.json'), 'w') as f:
            json.dump(fig_regional_rating.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
        print("Saved: ratings/regional_sales_by_rating_interactive.json")
    else:
        print("Skipping interactive Regional Sales by Rating plot: 'Rating' column has too many missing values after dropping NaNs.")

    # Business Question: What is the distribution of game ratings within the most popular genres?
    # Insight: Helps identify content trends and target audiences within successful genres.
    # Interactive Stacked Bar Chart for Rating Distribution per Genre
    if not df_rating_sales.empty:
        top_genres_list_for_rating = df_rating_sales['Genre'].value_counts().nlargest(10).index.tolist()
        df_top_genres_ratings_interactive = df_rating_sales[df_rating_sales['Genre'].isin(top_genres_list_for_rating)]

        if not df_top_genres_ratings_interactive.empty:
            genre_rating_counts_for_plotly = df_top_genres_ratings_interactive.groupby(['Genre', 'Rating']).size().reset_index(name='count')

            fig_genre_rating_stacked = px.bar(
                genre_rating_counts_for_plotly,
                x='Genre',
                y='count',
                color='Rating',
                title='Rating Distribution within Top 10 Genres (Interactive Stacked Bar)',
                labels={'Genre': 'Genre', 'count': 'Number of Games'},
                hover_data={'Rating': True, 'count': True},
                height=600
            )
            fig_genre_rating_stacked.update_layout(xaxis_tickangle=-45)
            with open(os.path.join(rating_plots_dir_html, 'rating_distribution_per_genre_interactive.json'), 'w') as f:
                json.dump(fig_genre_rating_stacked.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
            print("Saved: ratings/rating_distribution_per_genre_interactive.json")
        else:
            print("Skipping interactive Rating Distribution per Genre plot: No data for top genres with ratings.")
    else:
        print("Skipping interactive Rating Distribution per Genre plot: 'Rating' column has too many missing values after dropping NaNs.")


    # Release Quarter/Month Analysis Interactive Plots
    print("\n--- Generating Interactive Plots for Release Seasonality ---")
    seasonality_plots_dir_html = os.path.join(plots_dir, 'seasonality')
    os.makedirs(seasonality_plots_dir_html, exist_ok=True)

    # Business Question: What is the overall trend of global video game sales over time?
    # Insight: Shows the industry's historical growth or decline.
    yearly_global_sales_interactive = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
    fig_yearly_sales_interactive = px.line(
        yearly_global_sales_interactive,
        x='Year_of_Release',
        y='Global_Sales',
        title='Total Global Sales Trend by Year (Interactive)',
        labels={'Global_Sales': 'Total Global Sales (Millions)'},
        hover_data={'Year_of_Release': True, 'Global_Sales': ':.2f'}
    )
    fig_yearly_sales_interactive.update_layout(xaxis_title="Year of Release", yaxis_title="Total Sales (Millions)")
    with open(os.path.join(seasonality_plots_dir_html, 'yearly_global_sales_trend_interactive.json'), 'w') as f:
        json.dump(fig_yearly_sales_interactive.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: seasonality/yearly_global_sales_trend_interactive.json")

    # Business Question: How do average critic and user scores vary by genre?
    # Insight: Helps understand general quality perception across different game types.
    # Interactive Box Plots of Scores/Sales by Genre
    print("\n--- Generating Interactive Box Plots for Scores/Sales by Genre ---")
    genre_metrics_plots_dir_html = os.path.join(plots_dir, 'genre_metrics')
    os.makedirs(genre_metrics_plots_dir_html, exist_ok=True)

    # Interactive Box plot for Critic Score by Genre
    fig_critic_score_genre_box = px.box(
        df,
        x='Genre',
        y='Critic_Score',
        title='Distribution of Critic Scores by Genre (Interactive)',
        labels={'Critic_Score': 'Critic Score (out of 100)'},
        hover_data={'Name': True, 'Platform': True, 'Global_Sales': ':.2f'}
    )
    fig_critic_score_genre_box.update_layout(xaxis_tickangle=-45)
    with open(os.path.join(genre_metrics_plots_dir_html, 'critic_score_by_genre_boxplot_interactive.json'), 'w') as f:
        json.dump(fig_critic_score_genre_box.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: genre_metrics/critic_score_by_genre_boxplot_interactive.json")

    # Interactive Box plot for User Score by Genre (using scaled score)
    if 'User_Score_Scaled' in df.columns:
        fig_user_score_genre_box = px.box(
            df,
            x='Genre',
            y='User_Score_Scaled',
            title='Distribution of User Scores (Scaled) by Genre (Interactive)',
            labels={'User_Score_Scaled': 'User Score (Scaled to 100)'},
            hover_data={'Name': True, 'Platform': True, 'Global_Sales': ':.2f'}
        )
        fig_user_score_genre_box.update_layout(xaxis_tickangle=-45)
        with open(os.path.join(genre_metrics_plots_dir_html, 'user_score_by_genre_boxplot_interactive.json'), 'w') as f:
            json.dump(fig_user_score_genre_box.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
        print("Saved: genre_metrics/user_score_by_genre_boxplot_interactive.json")
    else:
        print("Skipping Interactive User Score by Genre Boxplot: 'User_Score_Scaled' not available.")

    # Interactive Box plot for Global Sales by Genre (using log scale)
    fig_global_sales_genre_box = px.box(
        df,
        x='Genre',
        y='Global_Sales',
        title='Distribution of Global Sales by Genre (Interactive, Log Scale)',
        labels={'Global_Sales': 'Global Sales (Millions, Log Scale)'},
        log_y=True, # Apply log scale to Y-axis
        hover_data={'Name': True, 'Platform': True, 'Critic_Score': ':.1f'}
    )
    fig_global_sales_genre_box.update_layout(xaxis_tickangle=-45)
    with open(os.path.join(genre_metrics_plots_dir_html, 'global_sales_by_genre_boxplot_interactive.json'), 'w') as f:
        json.dump(fig_global_sales_genre_box.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: genre_metrics/global_sales_by_genre_boxplot_interactive.json")

    # Top 25 Games by Global Sales with Critic and User Scores
    print("\n--- Generating Interactive Top 25 Games by Global Sales and Scores Plot ---")
    best_games_plots_dir_html = os.path.join(plots_dir, 'best_games')
    os.makedirs(best_games_plots_dir_html, exist_ok=True)

    # Business Question: Which games are the highest-selling globally, and how do their critic and user scores compare?
    # Insight: Identifies top commercial successes and the alignment of critical/user reception with sales.
    # Data preparation as per user's notebook code
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

    ucs_name = df.groupby('Name').agg({
        'Critic_Score': 'mean',
        'User_Score': 'mean',
        'Global_Sales': 'sum'
    }).reset_index()

    ucs_name['Critic_Score'] = ucs_name['Critic_Score'] / 10

    ucs_name.dropna(subset=['Critic_Score', 'User_Score', 'Global_Sales'], inplace=True)

    t25_name = ucs_name.sort_values('Global_Sales', ascending=False).head(25).reset_index(drop=True)

    fig_top_sales_scores = go.Figure()

    fig_top_sales_scores.add_trace(go.Bar(
        x=t25_name['Name'],
        y=t25_name['Global_Sales'],
        name='Global Sales',
        marker_color='rgba(0,0,0,0)',
        marker_line_color='blue',
        marker_line_width=1,
        opacity=0.8,
        hoverinfo='x+y',
        hovertemplate='<b>%{x}</b><br>Global Sales: %{y:.2f}M<extra></extra>'
    ))

    fig_top_sales_scores.add_trace(go.Scatter(
        x=t25_name['Name'],
        y=t25_name['Critic_Score'],
        name='Critic Score',
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        yaxis='y2',
        hoverinfo='x+y',
        hovertemplate='<b>%{x}</b><br>Critic Score: %{y:.1f}<extra></extra>'
    ))

    fig_top_sales_scores.add_trace(go.Scatter(
        x=t25_name['Name'],
        y=t25_name['User_Score'],
        name='User Score',
        mode='markers',
        marker=dict(color='green', size=10, symbol='circle'),
        yaxis='y2',
        hoverinfo='x+y',
        hovertemplate='<b>%{x}</b><br>User Score: %{y:.1f}<extra></extra>'
    ))

    fig_top_sales_scores.update_layout(
        title_text='Top 25 Games by Global Sales with Critic and User Scores (Interactive)',
        xaxis_title='Game Name',
        yaxis=dict(
            title='Global Sales (Millions)',
            title_font=dict(color='blue'),
            tickfont=dict(color='blue'),
            side='left'
        ),
        yaxis2=dict(
            title='Score (out of 10)',
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            overlaying='y',
            side='right',
            range=[0, 10]
        ),
        hovermode='x unified',
        xaxis_tickangle=-45,
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        height=700 # Increased height for better readability
    )

    with open(os.path.join(best_games_plots_dir_html, 'top_25_global_sales_and_scores_interactive.json'), 'w') as f:
        json.dump(fig_top_sales_scores.to_dict(), f, cls=NumpyEncoder) # Use custom encoder
    print("Saved: best_games/top_25_global_sales_and_scores_interactive.json")


def create_dashboard(summary_stats_html, plots_dir='plots/html'):
    """
    Creates a single HTML dashboard file that embeds the interactive Plotly graphs.
    This dashboard provides a consolidated view of key insights.

    Args:
        summary_stats_html (str): HTML string containing summary statistics.
        plots_dir (str): Directory where interactive plots are saved and dashboard will be saved.
    """
    print(f"\n--- Creating Interactive Dashboard in {plots_dir} ---")
    dashboard_path = os.path.join(plots_dir, 'dashboard.html')

    # Paths to individual interactive plots relative to the dashboard.html
    plot_files = {
        'regional_sales_by_genre': 'regional_sales_by_genre_interactive.json',
        'regional_sales_trends': 'regional_sales_trends_interactive.json',
        'sales_by_platform': 'sales_by_platform_interactive.json',
        'sales_trends': 'sales_trends_interactive.json',
        'critic_vs_user_score': 'critic_vs_user_score_scatter_enhanced_interactive.json',
        'publisher_sales_trends': 'publishers_developers/publisher_sales_trends_interactive.json',
        'avg_sales_per_publisher': 'publishers_developers/avg_sales_per_publisher_interactive.json',
        'regional_sales_by_rating': 'ratings/regional_sales_by_rating_interactive.json',
        'rating_distribution_per_genre_interactive': 'ratings/rating_distribution_per_genre_interactive.json',
        'yearly_global_sales_trend_interactive': 'seasonality/yearly_global_sales_trend_interactive.json',
        'critic_score_by_genre_boxplot_interactive': 'genre_metrics/critic_score_by_genre_boxplot_interactive.json',
        'user_score_by_genre_boxplot_interactive': 'genre_metrics/user_score_by_genre_boxplot_interactive.json',
        'global_sales_by_genre_boxplot_interactive_log': 'genre_metrics/global_sales_by_genre_boxplot_interactive.json', # Changed to .json
        'top_25_global_sales_and_scores_interactive': 'best_games/top_25_global_sales_and_scores_interactive.json',
        'top_25_publishers_sales_scores_interactive': 'publishers_developers/top_25_publishers_sales_scores_interactive.json'
    }

    # Read content of each Plotly JSON file
    plot_json_data = {}
    for key, filename in plot_files.items():
        file_path = os.path.join(plots_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    plot_json_data[key] = json.load(f)
                except json.JSONDecodeError as e:
                    plot_json_data[key] = None
                    print(f"Warning: Error decoding JSON from {filename}: {e}. Plot will not be displayed.")
        else:
            plot_json_data[key] = None
            print(f"Warning: {filename} not found for dashboard. Plot will not be displayed.")

    # Generate Plotly rendering script calls
    plotly_script_calls = ""
    for key, json_data in plot_json_data.items():
        if json_data:
            div_id = f"plotly-div-{key}"
            # Plotly.js expects data and layout as separate arguments
            data_json_str = json.dumps(json_data.get('data', []))
            layout_json_str = json.dumps(json_data.get('layout', {}))
            plotly_script_calls += f"""
            var {key}_data = {data_json_str};
            var {key}_layout = {layout_json_str};
            Plotly.newPlot('{div_id}', {key}_data, {key}_layout, {{responsive: true}});
            """
        else:
            plotly_script_calls += f"""
            document.getElementById('plotly-div-{key}').innerHTML = "<p class='text-red-500'>Error loading plot: {key}.</p>";
            """

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
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script> <!-- Load Plotly.js once -->
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
            ul.insight-list {{
                list-style: none; /* Remove default bullet */
                padding-left: 0;
            }}
            ul.insight-list li {{
                position: relative;
                padding-left: 1.5em; /* Space for custom bullet */
                margin-bottom: 0.5em;
            }}
            ul.insight-list li::before {{
                content: '🎮'; /* Custom emoji bullet */
                position: absolute;
                left: 0;
                color: #4F46E5; /* A distinct color */
                font-size: 1.2em;
                line-height: 1;
            }}
            /* Table styling for summary statistics */
            .table-auto {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 1rem;
            }}
            .table-auto th, .table-auto td {{
                border: 1px solid #e5e7eb;
                padding: 0.75rem;
                text-align: left;
            }}
            .table-auto th {{
                background-color: #f9fafb;
                font-weight: 600;
                color: #374151;
            }}
            .table-auto tbody tr:nth-child(odd) {{
                background-color: #f9fafb;
            }}
            .table-auto tbody tr:hover {{
                background-color: #f3f4f6;
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
                    <h2 class="card-title">Dataset Overview: Summary Statistics</h2>
                    {summary_stats_html}
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Regional Sales by Genre</h2>
                    <div id="plotly-div-regional_sales_by_genre" class="plotly-graph-div"></div>
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Regional and Global Sales Trends Over Time</h2>
                    <div id="plotly-div-regional_sales_trends" class="plotly-graph-div"></div>
                </div>

                <div class="card">
                    <h2 class="card-title">Total Global Sales by Platform</h2>
                    <div id="plotly-div-sales_by_platform" class="plotly-graph-div"></div>
                </div>

                <div class="card">
                    <h2 class="card-title">Global Sales Trends Over Time</h2>
                    <div id="plotly-div-sales_trends" class="plotly-graph-div"></div>
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Critic Score vs User Score (Scaled for Comparison, Sized by Global Sales)</h2>
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Enhanced Scatter Plot (with opacity for density):</h3>
                    <div id="plotly-div-critic_vs_user_score" class="plotly-graph-div"></div>
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Publisher and Developer Impact</h2>
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Publisher Sales Trends Over Time:</h3>
                    <div id="plotly-div-publisher_sales_trends" class="plotly-graph-div"></div>
                    <h3 class="text-lg font-semibold text-gray-700 mt-4 mb-2">Average Sales per Game by Publisher:</h3>
                    <div id="plotly-div-avg_sales_per_publisher" class="plotly-graph-div"></div>
                    <h3 class="text-lg font-semibold text-gray-700 mt-4 mb-2">Top 25 Publishers by Global Sales with Average Critic and User Scores:</h3>
                    <div id="plotly-div-top_25_publishers_sales_scores_interactive" class="plotly-graph-div"></div>
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Game Rating Analysis</h2>
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Regional Sales by Game Rating:</h3>
                    <div id="plotly-div-regional_sales_by_rating" class="plotly-graph-div"></div>
                    <h3 class="text-lg font-semibold text-gray-700 mt-4 mb-2">Rating Distribution within Top 10 Genres:</h3>
                    <div id="plotly-div-rating_distribution_per_genre_interactive" class="plotly-graph-div"></div>
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Genre Performance Metrics</h2>
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Distribution of Critic Scores by Genre:</h3>
                    <div id="plotly-div-critic_score_by_genre_boxplot_interactive" class="plotly-graph-div"></div>
                    <h3 class="text-lg font-semibold text-gray-700 mt-4 mb-2">Distribution of User Scores (Scaled) by Genre:</h3>
                    <div id="plotly-div-user_score_by_genre_boxplot_interactive" class="plotly-graph-div"></div>
                    <h3 class="text-lg font-semibold text-gray-700 mt-4 mb-2">Distribution of Global Sales by Genre (Log Scale):</h3>
                    <div id="plotly-div-global_sales_by_genre_boxplot_interactive_log" class="plotly-graph-div"></div>
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Overall Sales Trends</h2>
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Total Global Sales Trend by Year:</h3>
                    <div id="plotly-div-yearly_global_sales_trend_interactive" class="plotly-graph-div"></div>
                </div>

                <div class="card col-span-2">
                    <h2 class="card-title">Top Performing Games</h2>
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Top 25 Games by Global Sales with Critic and User Scores:</h3>
                    <div id="plotly-div-top_25_global_sales_and_scores_interactive" class="plotly-graph-div"></div>
                </div>

            </div>

            <footer class="text-center text-gray-500 mt-8 text-sm">
                <p>&copy; 2023 Video Game Sales Analysis. All rights reserved.</p>
            </footer>
        </div>
        <script type="text/javascript">
            // This script will render all Plotly graphs dynamically
            {plotly_script_calls}
        </script>
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
    os.makedirs(os.path.join(STATIC_PLOTS_DIR, 'ratings'), exist_ok=True) # Corrected line
    os.makedirs(os.path.join(STATIC_PLOTS_DIR, 'seasonality'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_PLOTS_DIR, 'genre_metrics'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_PLOTS_DIR, 'best_games'), exist_ok=True)

    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'publishers_developers'), exist_ok=True)
    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'ratings'), exist_ok=True)
    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'seasonality'), exist_ok=True)
    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'genre_metrics'), exist_ok=True)
    os.makedirs(os.path.join(HTML_PLOTS_DIR, 'best_games'), exist_ok=True)


    df = load_processed_data(PROCESSED_DATA_PATH)
    if df is not None:
        # Generate summary statistics HTML
        summary_stats_html_content = generate_summary_statistics(df)
        
        create_static_plots(df, STATIC_PLOTS_DIR)
        create_interactive_plots(df, HTML_PLOTS_DIR)
        # Pass the summary statistics HTML to the dashboard creation function
        create_dashboard(summary_stats_html_content, HTML_PLOTS_DIR)
