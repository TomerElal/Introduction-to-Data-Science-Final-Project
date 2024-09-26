import argparse

from algorithms.nlp import *
from algorithms.tf_idf import *
from algorithms.k_means import *
from utils.preprocess import *


def k_means_execute():
    # Apply 1st K-means clustering - Numeric features
    k_means_features = PostFields.NUMERIC_COLS.value
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - Numeric Columns',
          k=2, func=euclidean_distance, update_centroid_func=mean_centroid_func, type='numeric')

    # Apply 2nd K-means clustering - Dummies features
    k_means_features = {PostFields.HAS_VIDEO.value, PostFields.HAS_IMAGE.value}
    for col in df.columns:
        if col.startswith(PostFields.POST_MAIN_SUBJECT.value) or col.startswith(PostFields.POST_MAIN_FEELING.value):
            k_means_features.add(col)
    k_means_features = list(k_means_features)
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - Categorical Columns',
          k=2, func=binary_distance, update_centroid_func=bit_count_centroid_func, type='categorical')


def nlp_execute(documents):
    tokens = [word.lower() for doc in documents for word in doc.split()]
    create_log_info_and_plot(tokens)
    create_word_cloud(tokens)


def tf_idf_execute(documents, post_to_predict=None):
    vocab_list = list(set(create_vocab_set(documents) + create_vocab_set([post_to_predict])))
    documents = documents + [post_to_predict] if post_to_predict else documents

    idf_dict = idf(vocab_list, documents)
    tf_idf_dict = generate_tf_idf_values(documents, vocab_list, idf_dict)

    if post_to_predict:
        similar_docs = get_similar_documents(tf_idf_dict, post_to_predict)
        plot_similar_documents(similar_docs, df, post_to_predict)
        plot_interactive_similar_documents(similar_docs, df, post_to_predict)

    return tf_idf_dict


if __name__ == "__main__":
    post_to_predict = "Im happy to share that I am starting a new job as a Machine Learning Engineer at Google !!"
    parser = argparse.ArgumentParser(description="Process some LinkedIn posts.")
    parser.add_argument('--post_to_pred', type=str, default=post_to_predict, help='Post content to predict')
    args = parser.parse_args()

    file_path = 'data_mining/Linkedin_Posts_withGPT.csv'
    df = preprocess_data(load_data(file_path))
    recap_documents = df[PostFields.CONTENT_FIRST_LINE.value].tolist()
    full_documents = df[PostFields.POST_CONTENT.value].tolist()

    # Calling all algorithms executions
    k_means_execute()
    print("Done Kmeans")
    nlp_execute(full_documents)
    print("Done NLP")
    tf_idf_execute(recap_documents, post_to_predict)
    print("Done TF-IDF")
