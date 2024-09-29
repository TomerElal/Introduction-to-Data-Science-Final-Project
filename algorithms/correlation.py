import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    # plt.show()


def evaluate_and_plot_corr_for_all_features_together(df):
    temp_df = df
    temp_df['PostRating'] = temp_df.apply(engagement_rating, axis=1)
    plot_pearson_correlation_with_post_rating(df, engagement_rating.__name__)


def evaluate_and_plot_corr_per_categorical_group(df):
    has_image_video_data = {
        "Feature": ["HasImage", "HasVideo", "NoImage", "NoVideo"],
        "Average NumShares": [
            df[df["HasImage"] == 1]["NumShares"].mean(),
            df[df["HasVideo"] == 1]["NumShares"].mean(),
            df[df["HasImage"] == 0]["NumShares"].mean(),
            df[df["HasVideo"] == 0]["NumShares"].mean()
        ]
    }
    has_image_video_df = pd.DataFrame(has_image_video_data)

    post_main_subject_data = {
        "Feature": [],
        "Average PostRating": []
    }

    post_main_subject_features = [col for col in df.columns if 'PostMainSubject' in col]
    for feature in post_main_subject_features:
        if feature not in post_main_subject_data["Feature"]:
            avg_rating = df[df[feature] == 1]["PostRating"].mean()
            post_main_subject_data["Feature"].append(feature)
            post_main_subject_data["Average PostRating"].append(avg_rating)

    post_main_subject_df = pd.DataFrame(post_main_subject_data)

    post_main_feeling_data = {
        "Feature": [],
        "Average PostRating": []
    }

    post_main_feeling_features = [col for col in df.columns if 'PostMainFeeling' in col]
    for feature in post_main_feeling_features:
        if feature not in post_main_feeling_data["Feature"]:
            avg_rating = df[df[feature] == 1]["PostRating"].mean()
            post_main_feeling_data["Feature"].append(feature)
            post_main_feeling_data["Average PostRating"].append(avg_rating)

    post_main_feeling_df = pd.DataFrame(post_main_feeling_data)

    # Plotting
    for data_df, title in zip([has_image_video_df, post_main_subject_df, post_main_feeling_df],
                              ["HasImage and HasVideo", "PostMainSubject Features", "PostMainFeeling Features"]):

        data_df['Feature'] = data_df['Feature'].apply(lambda x: x.split('_')[-1])

        by_y_axis_name = 'PostRating'
        if "Has" in title:
            by_y_axis_name = 'NumShares'

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Feature', y=f'Average {by_y_axis_name}', data=data_df, palette='viridis')
        plt.title(f'Average {by_y_axis_name} by {title}', fontsize=16)
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel(f'Average {by_y_axis_name}', fontsize=14)
        plt.xticks(rotation=45)
        plt.axhline(0, color='gray', linestyle='--')
        plt.tight_layout()

        # Save the plot
        plot_file_path = f'plots/pearson_correlation_plots/average_{by_y_axis_name.lower()}_by_{title.replace(" ", "_").lower()}.png'
        plt.savefig(plot_file_path)
        plt.clf()


def evaluate_and_plot_corr_per_numeric_feature(df):
    temp_df = df.copy()

    for feature in temp_df.columns[3:15]:
        if (feature.startswith('Has') or 'Comments' in feature
                or 'Shares' in feature or 'Reactions' in feature
                or 'Followers' in feature or 'Hashtags' in feature):
            continue
        plt.figure(figsize=(10, 8))

        # Scatter plot of feature vs. PostRating
        plt.scatter(temp_df[feature], temp_df['PostRating'], alpha=0.6, label='Data Points')

        # Calculate the gradient (line of best fit)
        if len(temp_df[feature]) > 1:
            coefficients = np.polyfit(temp_df[feature], temp_df['PostRating'], 1)  # Linear fit
            trend_line = np.polyval(coefficients, temp_df[feature])
            plt.plot(temp_df[feature], trend_line, color='red', linestyle='--', label='Trend Line')

            # Calculate Pearson correlation
            correlation_value = np.corrcoef(temp_df[feature], temp_df['PostRating'])[0, 1]

        # Setting the title and labels
        plt.title(f"Scatter Plot of {feature} vs PostRating", fontsize=18)
        plt.ylabel("PostRating", fontsize=16)
        plt.xlabel(feature, fontsize=16)
        plt.ylim(temp_df['PostRating'].min() - 0.1, temp_df['PostRating'].max() + 0.1)  # Set y-limits
        plt.axhline(0, color='gray', linewidth=1.5, linestyle='--')

        # Show correlation value in the title
        plt.legend()
        plt.text(0.05, 0.95, f'Pearson Corr: {correlation_value:.2f}', transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='top')

        plt.tight_layout()

        # Save the plot
        plot_file_path = f'plots/pearson_correlation_plots/correlation_{feature}.png'
        plt.savefig(plot_file_path)
        plt.clf()


def plot_kmeans_numeric_correlations(df, clusters, features):
    cluster_avg_post_ratings = []

    for cluster in clusters:
        cluster_df = df.iloc[[point[-1] for point in cluster]]
        avg_post_rating = cluster_df['PostRating'].mean()
        cluster_avg_post_ratings.append(avg_post_rating)

    best_cluster_idx = np.argmax(cluster_avg_post_ratings)
    best_cluster = clusters[best_cluster_idx]

    for feature in features:
        plt.figure(figsize=(10, 6))
        feature_values = np.array([df.at[point[-1], feature] for point in best_cluster])
        post_rating_values = np.array([df.at[point[-1], 'PostRating'] for point in best_cluster])

        plt.scatter(feature_values, post_rating_values, label=f'Best Cluster {best_cluster_idx + 1}', alpha=0.6)

        if len(feature_values) > 1:
            coefficients = np.polyfit(feature_values, post_rating_values, 1)  # Linear fit
            trend_line = np.polyval(coefficients, feature_values)
            plt.plot(feature_values, trend_line, color='red', linestyle='--', label='Gradient')

        plt.title(f'Correlation of {feature} with PostRating - Best Cluster ({best_cluster_idx + 1})')
        plt.xlabel(f'{feature} Values')
        plt.ylabel('PostRating Values')
        plt.legend()

        plot_file_path = f'plots/kmeans_plots/kmeans_numeric_correlation_plots/best_cluster_correlations/kmeans_correlation_{feature}.png'
        plt.savefig(plot_file_path)
        plt.clf()

        # plt.show()


def plot_kmeans_categorical_correlations(df, clusters, features, group_name):
    cluster_avg_post_ratings = []

    # Calculate average PostRating for each cluster
    for cluster in clusters:
        cluster_df = df.iloc[[point[-1] for point in cluster]]
        avg_post_rating = cluster_df['PostRating'].mean()
        cluster_avg_post_ratings.append(avg_post_rating)

    best_cluster_idx = np.argmax(cluster_avg_post_ratings)

    cluster_idx, cluster = best_cluster_idx, clusters[best_cluster_idx]
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

    plt.title(f'Average PostRating by Categorical Features - Best Cluster ({cluster_idx + 1})')
    plt.xlabel('Feature Values')
    plt.ylabel('Average PostRating')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)

    plot_file_path = f'plots/kmeans_plots/kmeans_categorical_correlation_plots/kmeans_correlation_{group_name}_best_cluster.png'
    plt.savefig(plot_file_path)

    plt.clf()

    # plt.show()
