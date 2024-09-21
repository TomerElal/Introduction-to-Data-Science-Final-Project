import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import *


def create_vocab_set(sentences):
    if not sentences:
        return []
    uniq_words = set()
    for sentence in sentences:
        for word in sentence.split(" "):
            word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
            if word.isalnum() and word.lower() not in uniq_words:  # Check if the word contains only alphabetic characters
                uniq_words.add(word.lower())
    return list(uniq_words)


def tf(word, document):
    word = word.lower()
    words_in_document = []
    for w in document.split(" "):
        w = re.sub(r'[^\w\s]', '', w)  # Remove punctuation
        if w.isalnum():  # Check if the word contains only alphabetic characters
            words_in_document.append(w.lower())
    count_appearances = words_in_document.count(word)
    return count_appearances / len(words_in_document)


def idf(word, documents):
    documents_no_rep = list(set(documents))
    number_of_documents_with_word = sum([1 for document in documents_no_rep if
                                         word.lower() in document.lower()])
    if not number_of_documents_with_word:
        return 1
    return math.log10(len(documents) / number_of_documents_with_word)


def tfidf(word, document, documents):
    return tf(word, document) * idf(word, documents)


def generate_tf_idf_values(documents, uniq_words):
    """
    Returns: A dict size
    key=#docs, value=list size #uniq_words each entry is tf-idf value(word, i'th doc, docs)
    """
    tf_idf_dict = {}
    for i, doc in enumerate(documents):
        print(f"process doc number {i} of {len(documents)}")
        tf_idf_dict[doc] = []
        for word in uniq_words:
            tf_idf_dict[doc].append(tfidf(word, doc.lower(), documents))
    return tf_idf_dict


def get_similar_documents(doc_dict, key, top_n=3):
    # Extract the TF-IDF values for the given key
    target_doc_tfidf = np.array(doc_dict[key]).reshape(1, -1)

    # Calculate cosine similarities between the target document and all other documents
    similarities = {}
    for doc_key, tfidf_values in doc_dict.items():
        if doc_key != key:  # Skip the target document itself
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


def plot_similar_documents(similar_docs, df):
    doc_keys, similarities = zip(*similar_docs)
    wrapped_keys = [wrap_text(key, 15) for key in doc_keys]

    post_ratings = []
    for key in doc_keys:
        if key in df.index:
            post_ratings.append(df.loc[key, 'PostRating'])
        else:
            post_ratings.append(0)

    plt.figure(figsize=(12, 8))

    # Create the first subplot for cosine similarities
    ax1 = plt.subplot(211)  # First subplot
    ax1.bar(wrapped_keys, similarities, color='blue')
    ax1.set_title('Top 3 Similar Documents', fontsize=18)
    ax1.set_ylabel('Cosine Similarity', fontsize=16)
    ax1.set_ylim(0, 1)  # Cosine similarity ranges from 0 to 1

    for index, value in enumerate(similarities):
        ax1.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=14)

    ax1.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid for better readability

    # Create the second subplot for PostRating values
    ax2 = plt.subplot(212)  # Second subplot
    ax2.bar(wrapped_keys, post_ratings, color='orange')
    ax2.set_title('PostRating of Similar Documents', fontsize=18)
    ax2.set_ylabel('PostRating', fontsize=16)
    ax2.set_ylim(min(post_ratings) - 0.1, max(post_ratings) + 0.1)  # Set limits based on PostRating values

    for index, value in enumerate(post_ratings):
        ax2.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=14)

    plt.tight_layout()
    # Save the plots
    plot_file_path = f'plots/similar_docs_prediction_with_their_ratings.png'
    plt.savefig(plot_file_path)

    plt.show()


def plot_all_top_tfidf_words(cnt_table):
    max_tfidf_values = {word: max(tfidf_vals) for word, tfidf_vals in cnt_table.items()}
    sorted_words = sorted(max_tfidf_values.items(), key=lambda x: x[1], reverse=True)
    words, tfidf_values = zip(*sorted_words)

    # Limit the number of top tf-idf words for better visibility
    limit = 20
    words = words[:limit]
    tfidf_values = tfidf_values[:limit]

    # Create the bar plot
    plt.figure(figsize=(12, 8))
    plt.barh(words, tfidf_values, color='skyblue')
    plt.xlabel('Maximum TF-IDF Value', fontsize=22)
    plt.title('Top Words by Maximum TF-IDF Value', fontsize=22)
    plt.gca().invert_yaxis()
    plt.tick_params(axis='both', labelsize=15)

    # Save the plots
    plot_file_path = f'plots/top_tf_idf_words.png'
    plt.savefig(plot_file_path)

    plt.show()
