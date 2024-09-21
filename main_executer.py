from algorithms.nlp import *
from algorithms.tf_idf import *
from algorithms.k_means import *
from utils.preprocess import *


def k_means_execute():
    # Apply 1st K-means clustering - Numeric features
    k_means_features = PostFields.NUMERIC_COLS.value
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - Numeric Columns',
          k=3, func=euclidean_distance, update_centroid_func=mean_centroid_func)

    # Apply 2nd K-means clustering - Dummies features
    k_means_features = {PostFields.HAS_VIDEO.value, PostFields.HAS_IMAGE.value}
    for col in df.columns:
        if col.startswith(PostFields.POST_MAIN_SUBJECT.value) or col.startswith(PostFields.POST_MAIN_FEELING.value):
            k_means_features.add(col)
    k_means_features = list(k_means_features)
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    apply(df, k_means_features, plot_features, 'K-means Clustering - Categorical Columns',
          k=3, func=binary_distance, update_centroid_func=bit_count_centroid_func)


def post_rating_execute():
    evaluate_and_plot_corr_per_feature(df)
    # evaluate_and_plot_corr_for_all_features_together(df)


def nlp_execute(documents):
    tokens = [word.lower() for doc in documents for word in doc.split()]
    create_log_info_and_plot(tokens)
    create_word_cloud(tokens)


def tf_idf_execute(documents, post_to_predict=None):
    vocab_list = list(set(create_vocab_set(documents) + create_vocab_set([post_to_predict])))
    documents = documents + [post_to_predict] if post_to_predict else documents
    tf_idf_dict = generate_tf_idf_values(documents, vocab_list)

    similar_docs = get_similar_documents(tf_idf_dict, post_to_predict)
    plot_similar_documents(similar_docs)


if __name__ == "__main__":
    file_path = 'data_mining/Linkedin_Posts.csv'
    df = preprocess_data(load_data(file_path))
    documents = df[PostFields.CONTENT_FIRST_LINE.value].tolist()

    # Calling all algorithms executions
    #k_means_execute()
    #post_rating_execute()
    #nlp_execute(documents)
    tf_idf_execute(documents, "Omer")
