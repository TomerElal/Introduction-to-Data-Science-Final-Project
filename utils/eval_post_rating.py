import matplotlib.pyplot as plt
import seaborn as sns


def engagement_rating(row):
    """
    This function calculates the post rating based on the engagement metrics
    (reactions, comments, and shares) relative to the user's followers.
    """

    num_reactions_factor = 1
    num_comments_factor = 1
    num_shares_factor = 2

    engagement = (num_reactions_factor * row['NumReactions'] +
                  num_comments_factor * row['NumComments'] +
                  num_shares_factor * row['NumShares'])
    if row['NumFollowers'] > 0:  # Avoid division by zero
        return 100 * engagement / row['NumFollowers']
    return 0


def content_quality_rating(row):
    """
    This function evaluates the quality of the post based on the content features such as the number
    of words, punctuation, emojis, and hashtags.
    It considers that more engaging content (more words, emojis, hashtags) tends to go viral.
    """
    # Weight factors for each feature
    word_weight = 0.1
    punctuation_weight = 0.2
    emoji_weight = 0.3
    hashtag_weight = 0.4

    # Calculate content quality score
    return (word_weight * row['NumWords'] +
            punctuation_weight * row['NumPunctuation'] +
            emoji_weight * row['NumEmojis'] +
            hashtag_weight * row['NumHashtags'])


def multifactor_rating(row):
    """
    This function combines engagement, content quality,
    and media presence (image or video) to create a more comprehensive rating.
    """
    engagement_score = engagement_rating(row)
    content_quality_score = content_quality_rating(row)
    media_bonus = 1.5 if row['HasImage'] or row['HasVideo'] else 1.0  # Bonus for posts with media

    # Combine scores (adjust weights as necessary)
    return (0.4 * engagement_score +
            0.4 * content_quality_score +
            0.2 * media_bonus)


def get_all_eval_funcs():
    return [engagement_rating, content_quality_rating, multifactor_rating]


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
    temp_df = df  # do not want to change mainly df outside the func
    for eval_func in get_all_eval_funcs():
        # Calculate PostRating using the evaluation function
        temp_df['PostRating'] = temp_df.apply(eval_func, axis=1)

        plot_pearson_correlation_with_post_rating(df, eval_func.__name__)


def evaluate_and_plot_corr_per_feature(df):
    temp_df = df.copy()
    correlation_data = {feature: [] for feature in temp_df.columns[2:]}

    for eval_func in get_all_eval_funcs():
        temp_df['PostRating'] = temp_df.apply(eval_func, axis=1)
        for feature in temp_df.columns[2:]:
            correlation_value = temp_df[feature].corr(temp_df['PostRating'], method='pearson')
            correlation_data[feature].append(correlation_value)

    for feature in temp_df.columns[2:]:
        plt.figure(figsize=(10, 8))

        sns.barplot(x=[eval_func.__name__ for eval_func in get_all_eval_funcs()],
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

