import argparse

from algorithms.nlp import *
from algorithms.tf_idf import *
from algorithms.k_means import *
from utils.preprocess import *


def k_means_execute():
    # Apply 1st K-means clustering
    k_means_features = [PostFields.NUM_WORDS.value, PostFields.NUM_PUNCTUATION.value,
                        PostFields.NUM_LINKS.value, PostFields.NUM_LINE_BREAKS.value]
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    apply(df, k_means_features, plot_features, 'K-means-Clustering-Post-Structure',
          k=3, func=euclidean_distance, update_centroid_func=mean_centroid_func,
          type='numeric', sub_title='Post Structure')

    # Apply 2nd K-means clustering
    k_means_features = [PostFields.NUM_EMOJIS.value, PostFields.NUM_HASHTAGS.value]
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    apply(df, k_means_features, plot_features, 'K-means-Clustering-External-Columns',
          k=2, func=euclidean_distance, update_centroid_func=mean_centroid_func,
          type='numeric', sub_title='External Columns')

    # Apply 3rd K-means clustering - Dummies features
    k_means_features = [PostFields.HAS_VIDEO.value, PostFields.HAS_IMAGE.value]
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_SHARES.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - Visual Columns',
          k=3, func=binary_distance, update_centroid_func=bit_count_centroid_func,
          type='categorical', sub_title='Visual Columns')


def nlp_execute(documents):
    tokens = [word.lower() for doc in documents for word in doc.split()]
    create_log_info_and_plot(tokens)
    create_word_cloud(tokens)


def find_top_10_post_rating_avg(similar_docs):
    top_10_post_ratings = []
    for doc in similar_docs:
        matching_row = df[df['ContentFirstLine'] == doc[0]]
        top_10_post_ratings.append(matching_row['PostRating'].values[0])

    return round(sum(top_10_post_ratings) / len(top_10_post_ratings), 2)


def tf_idf_execute(documents, post_to_predict=None):
    vocab_list = list(set(create_vocab_set(documents) + create_vocab_set([post_to_predict])))
    documents = documents + [post_to_predict] if post_to_predict else documents

    idf_dict = idf(vocab_list, documents)
    tf_idf_dict = generate_tf_idf_values(documents, vocab_list, idf_dict)

    if post_to_predict:
        similar_docs = get_similar_documents(tf_idf_dict, post_to_predict, top_n=3)
        top_10_post_rating_avg = find_top_10_post_rating_avg(similar_docs)

        plot_similar_documents(similar_docs[:3], df, top_10_post_rating_avg)
        plot_interactive_similar_documents(similar_docs[:3], df, post_to_predict, top_10_post_rating_avg)

    return tf_idf_dict


def correlation_execute():
    evaluate_and_plot_corr_per_numeric_feature(df)
    evaluate_and_plot_corr_per_categorical_group(df)


if __name__ == "__main__":
    post_to_predict = "Im happy to share that I am starting a new job as a Machine Learning Engineer at Google !!"
    parser = argparse.ArgumentParser(description="Process some LinkedIn posts.")
    parser.add_argument('--post_to_pred', type=str, default=post_to_predict, help='Post content to predict')
    args = parser.parse_args()

    file_path = 'data_mining/Linkedin_Posts_withGPT.csv'
    df = preprocess_data(load_data(file_path))
    recap_documents = df[PostFields.CONTENT_FIRST_LINE.value].tolist()
    full_documents = df[PostFields.POST_CONTENT.value].tolist()

    df = load_data('data_mining/Linkedin_Posts.csv')

    # Calling all algorithms executions
    correlation_execute()
    print("Done Correlations")
    k_means_execute()
    print("Done Kmeans")
    nlp_execute(full_documents)
    print("Done NLP")
    tf_idf_execute(recap_documents, post_to_predict)
    print("Done TF-IDF")
