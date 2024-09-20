from utils.preprocess import *
from algorithms.k_means import *
from algorithms.tf_idf import *


def k_means_execute():
    # Apply 1st K-means clustering - Numeric features
    k_means_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    plot_features = [PostFields.NUM_WORDS.value, PostFields.NUM_SHARES.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - NumReactions, NumComments',
          k=3, func=euclidean_distance, update_centroid_func=mean_centroid_func)

    # # # Apply 2nd K-means clustering - Categorical features
    k_means_features = [PostFields.HAS_IMAGE.value, PostFields.HAS_VIDEO.value]
    plot_features = [PostFields.NUM_WORDS.value, PostFields.NUM_SHARES.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - HasImage, HasVideo',
          k=3, func=binary_distance, update_centroid_func=bit_count_centroid_func)

    # # Apply 3rd K-means clustering - Numeric Features
    k_means_features = [PostFields.NUM_HASHTAGS.value, PostFields.NUM_EMOJIS.value]
    plot_features = [PostFields.NUM_WORDS.value, PostFields.NUM_SHARES.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - NumHashtags, NumEmojis',
          k=3, func=euclidean_distance, update_centroid_func=mean_centroid_func)

    # # Apply 4th K-means clustering - Dummies features: TODO change for real ones.
    k_means_features = [PostFields.POST_MAIN_SUBJECT.value + "_a", PostFields.POST_MAIN_SUBJECT.value + "_d",
                        PostFields.POST_MAIN_FEELING.value + "_a", PostFields.POST_MAIN_FEELING.value + "_d"]
    plot_features = [PostFields.NUM_WORDS.value, PostFields.NUM_SHARES.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - PostMainSubject, PostMainFeeling',
          k=3, func=binary_distance, update_centroid_func=bit_count_centroid_func)

    # # Apply 5th K-means clustering - Numeric Cols
    k_means_features = PostFields.NUMERIC_COLS.value
    plot_features = [PostFields.NUM_WORDS.value, PostFields.NUM_SHARES.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - All Numeric Cols',
          k=3, func=euclidean_distance, update_centroid_func=mean_centroid_func)


def post_rating_execute():
    evaluate_and_plot_corr(df)


if __name__ == "__main__":
    file_path = 'Linkedin_Posts.csv'
    df = preprocess_data(load_data(file_path))

    k_means_execute()
    post_rating_execute()
