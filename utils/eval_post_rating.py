import matplotlib.pyplot as plt


def engagement_rating(row):
    """
    This function calculates the post rating based on the engagement metrics
    (reactions, comments, and shares) relative to the user's followers.
    """
    # Calculate engagement ratio
    engagement = row['NumReactions'] + row['NumComments'] + row['NumShares']
    if row['NumFollowers'] > 0:  # Avoid division by zero
        return engagement / row['NumFollowers']
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


def plot_correlation_with_post_rating(df, eval_func_name):
    plt.figure(figsize=(14, 10))

    # Calculate the correlation matrix excluding the first column
    correlation_matrix = df.iloc[:, 1:].corr()  # Exclude the first column
    postrating_corr = correlation_matrix.loc[:, 'PostRating']

    # Create a bar plot for the correlations with PostRating
    postrating_corr.drop('PostRating').plot(kind='bar', color='darkblue')  # Darker color

    plt.title(f"Correlation of Features to PostRating Value\n(using {eval_func_name})", fontsize=22)
    plt.ylabel("Correlation Coefficient", fontsize=22)
    plt.xticks(rotation=45)
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.show()


def evaluate_and_plot_corr(df):
    # Loop through each evaluation function
    for eval_func in get_all_eval_funcs():
        # Calculate PostRating using the evaluation function
        df['PostRating'] = df.apply(eval_func, axis=1)

        # Plot the heatmap
        plot_correlation_with_post_rating(df, eval_func.__name__)
