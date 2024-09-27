import math
import nltk
import re
import numpy as np
import plotly.graph_objects as go

from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import *
from nltk.corpus import words


nltk.download('words')
english_words = set(words.words())


def create_vocab_set(sentences):
    if not sentences:
        return []
    uniq_words = set()
    for sentence in sentences:
        for word in sentence.split(" "):
            word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
            if word.isalpha() and word.lower() not in uniq_words and word.lower() in english_words:
                uniq_words.add(word.lower())
    return list(uniq_words)


def tf(word, document):
    word = word.lower()
    words_in_document = []
    for w in document.split(" "):
        w = re.sub(r'[^\w\s]', '', w)  # Remove punctuation
        if w.isalpha():  # Check if the word contains only alphabetic characters
            words_in_document.append(w.lower())
    count_appearances = words_in_document.count(word)
    return count_appearances / len(words_in_document)


def idf(vocab_set, documents):
    idf_dict = {}
    documents_no_rep = list(set(documents))

    for word in vocab_set:
        number_of_documents_with_word = sum([1 for document in documents_no_rep if word.lower() in document.lower()])
        if not number_of_documents_with_word:
            idf_dict[word] = 1
            continue
        idf_value = math.log10(len(documents) / number_of_documents_with_word)
        idf_dict[word] = idf_value
    return idf_dict


def tfidf(word, document, idf_dict):
    return tf(word, document) * idf_dict.get(word, 0)  # Default IDF to 0 if not found


def generate_tf_idf_values(documents, uniq_words, idf_dict):
    tf_idf_dict = {}
    uniq_words = np.array(uniq_words)
    for i, doc in enumerate(documents):
        print(f"Processing document {i + 1} of {len(documents)}")
        words_in_document = preprocess_document(doc)
        word_counts = Counter(words_in_document)
        total_words = len(words_in_document)

        tf_values = np.array([word_counts[word] / total_words if word in word_counts else 0 for word in uniq_words])
        idf_values = np.array([idf_dict.get(word, 0) for word in uniq_words])
        tf_idf_values = tf_values * idf_values
        tf_idf_dict[doc] = tf_idf_values.tolist()

    return tf_idf_dict


def get_similar_documents(doc_dict, new_doc, top_n=3):
    # Extract the TF-IDF values for the given key
    target_doc_tfidf = np.array(doc_dict[new_doc]).reshape(1, -1)

    # Calculate cosine similarities between the target document and all other documents
    similarities = {}
    for doc_key, tfidf_values in doc_dict.items():
        if doc_key != new_doc:  # Skip the target document itself
            cosine_sim = cosine_similarity(target_doc_tfidf, np.array(tfidf_values).reshape(1, -1))[0][0]
            similarities[doc_key] = cosine_sim

    # Sort by similarity and get the top N
    similar_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]

    return similar_docs


def wrap_text(text, max_width):
    words = text.split()
    wrapped_lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_width:
            current_line += (word + " ")
        else:
            wrapped_lines.append(current_line.strip())
            current_line = word + " "

    wrapped_lines.append(current_line.strip())  # Add the last line
    return '\n'.join(wrapped_lines)


def preprocess_document(document):
    words = re.sub(r'[^\w\s]', '', document.lower()).split()
    return words


def plot_similar_documents(similar_docs, df, top_10_post_rating_avg):
    doc_keys, similarities = zip(*similar_docs)
    wrapped_keys = [wrap_text(key, 15) for key in doc_keys]

    post_ratings = []
    for key in doc_keys:
        if key in df['ContentFirstLine'].values:
            line_index = df[df['ContentFirstLine'] == key].index[0]
            post_ratings.append(df.loc[line_index, 'PostRating'])
        else:
            post_ratings.append(0)

    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(211)
    ax1.bar(wrapped_keys, similarities, color='blue')
    ax1.set_title(f'Top 3 Similar posts for given post with high probability achieving: {round(top_10_post_rating_avg, 2)} rating', fontsize=18)
    ax1.set_ylabel('Cosine Similarity', fontsize=16)
    ax1.set_ylim(0, 1)

    for index, value in enumerate(similarities):
        ax1.text(index, value + 0.05, f'{value:.2f}', ha='center', fontsize=14)

    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2 = plt.subplot(212)  # Second subplot
    ax2.bar(wrapped_keys, post_ratings, color='orange')
    ax2.set_title('PostRating of Similar Documents', fontsize=18)
    ax2.set_ylabel('PostRating', fontsize=16)
    ax2.set_ylim(0, max(post_ratings) + 1)

    for index, value in enumerate(post_ratings):
        ax2.text(index, value - 8, f'{value:.2f}', ha='center', fontsize=14)

    plt.tight_layout()
    # Save the plots
    plot_file_path = f'plots/tf_idf_plots/similar_docs_prediction_with_their_ratings.png'
    plt.savefig(plot_file_path)

    plt.show()


def plot_interactive_similar_documents(similar_docs, df, post_to_predict, top_10_post_rating_avg):
    doc_keys, similarities = zip(*similar_docs)

    post_contents = []
    for key in doc_keys:
        if key in df['ContentFirstLine'].values:
            line_index = df[df['ContentFirstLine'] == key].index[0]
            post_contents.append(df.loc[line_index, 'PostStart'])
        else:
            post_contents.append("Unknown Content")

    def wrap_label(text, width=50):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        lines.append(current_line.strip())
        return "<br>".join(lines)

    wrapped_post_contents = [wrap_label(content, width=15) for content in post_contents]

    post_ratings = []
    for key in doc_keys:
        if key in df['ContentFirstLine'].values:
            line_index = df[df['ContentFirstLine'] == key].index[0]
            post_ratings.append(df.loc[line_index, 'PostRating'])
        else:
            post_ratings.append(0)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=wrapped_post_contents,
            y=similarities,
            text=[f'{sim:.2f}' for sim in similarities],
            textposition='auto',
            marker_color='blue',
            name="Cosine Similarity",
            visible=True
        )
    )

    fig.add_trace(
        go.Bar(
            x=wrapped_post_contents,
            y=post_ratings,
            text=[f'{rating:.2f}' for rating in post_ratings],
            textposition='auto',
            marker_color='orange',
            name="PostRating",
            visible=False
        )
    )

    fig.update_layout(
        title=f"Interactive Cosine similarity & PostRatings<br>for: '{wrap_label(post_to_predict, 50)}'",
        title_x=0.5,
        title_y=0.95,
        title_font=dict(size=16),
        xaxis=dict(
            title='Post Content',
            tickangle=0,
            tickmode='array',
            tickvals=list(range(len(wrapped_post_contents))),
            ticktext=wrapped_post_contents
        ),
        yaxis=dict(title='Value'),
        width=1000,
        height=700,
        margin=dict(t=100),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=[
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Cosine Similarity",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="PostRating",
                        method="update"
                    )
                ],
                showactive=True,
                x=1.2,
                y=1.1
            )
        ]
    )

    fig.show()


def baseline_prediction(df, post_to_predict, numeric_cols):
    """
    Finds the top 3 similar documents according to norm2 distance of numeric columns and plots the result.

    :param df: DataFrame containing the documents and their numeric values
    :param post_to_predict: The post (document) for which we need to predict the similarity
    :param numeric_cols: List of numeric columns to use for norm2 distance calculation
    :return: Top 3 similar documents and their average PostRating
    """
    # Extract the numeric values of the post_to_predict
    if post_to_predict in df['ContentFirstLine'].values:
        post_index = df[df['ContentFirstLine'] == post_to_predict].index[0]
    else:
        print(f"Post '{post_to_predict}' not found in the DataFrame.")
        return None

    post_numeric_values = df.loc[post_index, numeric_cols].values

    # Calculate norm2 (Euclidean) distance between the post_to_predict and all other documents
    distances = []

    for idx, row in df.iterrows():
        if idx != post_index:  # Skip the post itself
            other_post_numeric_values = row[numeric_cols].values
            norm2_distance = np.linalg.norm(post_numeric_values - other_post_numeric_values)
            distances.append((idx, norm2_distance))

    # Sort by the smallest distances (most similar)
    distances = sorted(distances, key=lambda x: x[1])

    # Get the top 3 closest documents
    top_3_similar_docs = distances[:3]

    # Calculate the average PostRating for the top 3 similar documents
    top_3_post_ratings = [df.loc[idx, 'PostRating'] for idx, _ in top_3_similar_docs]
    top_3_post_rating_avg = np.mean(top_3_post_ratings)

    # Prepare data for plotting
    top_3_indices = [idx for idx, _ in top_3_similar_docs]
    top_3_distances = [dist for _, dist in top_3_similar_docs]
    top_3_titles = [df.loc[idx, 'ContentFirstLine'] for idx in top_3_indices]

    # Plotting the results
    plt.figure(figsize=(12, 8))

    # First subplot: Euclidean distances
    ax1 = plt.subplot(211)
    ax1.bar(top_3_titles, top_3_distances, color='blue')
    ax1.set_title(f'Top 3 Similar Posts to "{post_to_predict}" Based on Euclidean Distance', fontsize=18)
    ax1.set_ylabel('Norm2 Distance', fontsize=16)
    ax1.set_ylim(0, max(top_3_distances) * 1.2)

    for index, value in enumerate(top_3_distances):
        ax1.text(index, value + 0.05, f'{value:.2f}', ha='center', fontsize=14)

    # Second subplot: PostRating for top 3 similar posts
    ax2 = plt.subplot(212)
    ax2.bar(top_3_titles, top_3_post_ratings, color='orange')
    ax2.set_title(f'PostRatings of Top 3 Similar Posts', fontsize=18)
    ax2.set_ylabel('PostRating', fontsize=16)
    ax2.set_ylim(0, max(top_3_post_ratings) * 1.2)

    for index, value in enumerate(top_3_post_ratings):
        ax2.text(index, value + 0.05, f'{value:.2f}', ha='center', fontsize=14)

    plt.tight_layout()

    # Save the plot
    plot_file_path = f'plots/baseline_prediction/top_3_similar_posts_plot.png'
    plt.savefig(plot_file_path)

    plt.show()

    # Return the top 3 similar documents and the average PostRating
    return top_3_similar_docs, top_3_post_rating_avg
