import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.eval_post_rating import *


def plot_pearson_correlation_with_post_rating(df, eval_func_name):
    plt.figure(figsize=(16, 10))

    # Calculate the Pearson correlation matrix excluding the first column
    correlation_matrix = df.iloc[:, 2:].corr(method='pearson')  # Specify Pearson correlation
    postrating_corr = correlation_matrix.loc[:, 'PostRating']

    # Create a bar plot for the correlations with PostRating
    postrating_corr.drop('PostRating').plot(kind='bar', color='darkblue')  # Darker color

    plt.title(f"Pearson Correlation of Features to PostRating Value\n(using {eval_func_name})", fontsize=22)
    plt.ylabel("Correlation Coefficient", fontsize=22)
    # Rotate x-axis labels for readability
    plt.xticks(rotation=75, ha='right', fontsize=10)
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    # Save the plots
    plot_file_path = f'plots/pearson_correlation_with_postrating_{eval_func_name}.png'
    plt.savefig(plot_file_path)

    plt.show()


def evaluate_and_plot_corr_for_all_features_together(df):
    temp_df = df
    temp_df['PostRating'] = temp_df.apply(engagement_rating, axis=1)
    plot_pearson_correlation_with_post_rating(df, engagement_rating.__name__)


def evaluate_and_plot_corr_per_feature(df):
    temp_df = df.copy()
    correlation_data = {feature: [] for feature in temp_df.columns[2:]}

    temp_df['PostRating'] = temp_df.apply(engagement_rating, axis=1)
    for feature in temp_df.columns[2:]:
        correlation_value = temp_df[feature].corr(temp_df['PostRating'], method='pearson')
        correlation_data[feature].append(correlation_value)

    for feature in temp_df.columns[2:]:
        plt.figure(figsize=(10, 8))

        sns.barplot(x=[eval_func.__name__ for eval_func in [engagement_rating]],
                    y=correlation_data[feature], palette='viridis')

        plt.title(f"Pearson Correlation of {feature} with PostRating", fontsize=18)
        plt.ylabel("Pearson Correlation Coefficient", fontsize=16)
        plt.xlabel("Evaluation Functions", fontsize=16)
        plt.ylim(-1, 1)  # Set y-limits to focus on the correlation range

        # Enlarge x-axis labels
        plt.xticks(fontsize=14)

        # Draw a horizontal line at y=0
        plt.axhline(0, color='gray', linewidth=1.5, linestyle='--')

        for index, value in enumerate(correlation_data[feature]):
            plt.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=14)

        plt.tight_layout()

        plot_file_path = f'plots/pearson_correlation_plots/correlation_{feature}.png'
        plt.savefig(plot_file_path)

        # plt.show()


def plot_kmeans_numeric_correlations(df, clusters, features):
    for feature in features:
        for cluster_idx, cluster in enumerate(clusters):
            plt.figure(figsize=(10, 6))
            feature_values = np.array([df.at[point[-1], feature] for point in cluster])
            post_rating_values = np.array([df.at[point[-1], 'PostRating'] for point in cluster])

            # Plot the real points: x-axis = Feature values, y-axis = PostRating values
            plt.scatter(feature_values, post_rating_values, label=f'Cluster {cluster_idx + 1}', alpha=0.6)

            # Calculate the gradient (line of best fit)
            if len(feature_values) > 1:
                coefficients = np.polyfit(feature_values, post_rating_values, 1)  # Linear fit
                trend_line = np.polyval(coefficients, feature_values)
                plt.plot(feature_values, trend_line, color='red', linestyle='--', label='Gradient')

            plt.title(f'Correlation of {feature} with PostRating - Cluster {cluster_idx + 1}')
            plt.xlabel(f'{feature} Values')
            plt.ylabel('PostRating Values')
            plt.legend()

            plot_file_path = f'plots/kmeans_plots/kmeans_numeric_correlation_plots/cluster_{cluster_idx + 1}_correlation/kmeans_correlation_{feature}_cluster{cluster_idx + 1}.png'
            plt.savefig(plot_file_path)

            plt.show()


def plot_kmeans_categorical_correlations(df, clusters, features, group_name):
    cluster_avg_post_ratings = []

    for cluster in clusters:
        cluster_df = df.iloc[[point[-1] for point in cluster]]
        avg_post_rating = cluster_df['PostRating'].mean()
        cluster_avg_post_ratings.append(avg_post_rating)

    best_cluster_idx = np.argmax(cluster_avg_post_ratings)
    worst_cluster_idx = np.argmin(cluster_avg_post_ratings)

    # Plot for the best and worst clusters only
    selected_clusters = [(best_cluster_idx, clusters[best_cluster_idx]), (worst_cluster_idx, clusters[worst_cluster_idx])]

    for cluster_idx, cluster in selected_clusters:
        plt.figure(figsize=(12, 8))
        cluster_df = df.iloc[[point[-1] for point in cluster]]

        avg_post_ratings = {}

        for feature in features[:-1]:
            avg_post_rating = cluster_df.groupby(feature)['PostRating'].mean()
            avg_post_ratings[feature] = avg_post_rating

        feature_labels = []
        avg_ratings = []

        for feature, avg_rating in avg_post_ratings.items():
            if group_name in ["main_subject", "main_feeling"]:
                feature = feature.split("_")[-1].strip().lstrip().lower()
            feature_labels.extend([f'{feature}_{str(val)}' for val in avg_rating.index])
            avg_ratings.extend(avg_rating.values)

        sns.barplot(x=feature_labels, y=avg_ratings, alpha=0.8, palette='muted')

        cluster_type = 'Best' if cluster_idx == best_cluster_idx else 'Worst'
        plt.title(f'Average PostRating by Categorical Features - {cluster_type} Cluster ({cluster_idx + 1})')
        plt.xlabel('Feature Values')
        plt.ylabel('Average PostRating')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)

        plot_file_path = f'plots/kmeans_plots/kmeans_categorical_correlation_plots/kmeans_correlation_{group_name}_{cluster_type.lower()}_cluster{cluster_idx + 1}.png'
        plt.savefig(plot_file_path)

        plt.show()
