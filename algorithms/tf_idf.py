import math
import nltk
import re
import numpy as np

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


def plot_similar_documents(similar_docs, df, post_to_predict):
    doc_keys, similarities = zip(*similar_docs)
    wrapped_keys = [wrap_text(key, 15) for key in doc_keys]

    post_ratings = []
    for key in doc_keys:
        if key in df['ContentFirstLine'].values:
            line_index = df[df['ContentFirstLine'] == key].index[0]  # Get the first match
            post_ratings.append(df.loc[line_index, 'PostRating'])
        else:
            post_ratings.append(0)

    plt.figure(figsize=(12, 8))

    # Create the first subplot for cosine similarities
    ax1 = plt.subplot(211)  # First subplot
    ax1.bar(wrapped_keys, similarities, color='blue')
    ax1.set_title(f'Top 3 Similar Documents for "{post_to_predict}"', fontsize=18)
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
    ax2.set_ylim(0, max(post_ratings) + 0.5)  # Set limits based on PostRating values

    for index, value in enumerate(post_ratings):
        ax2.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=14)

    plt.tight_layout()
    # Save the plots
    plot_file_path = f'plots/tf_idf_plots/similar_docs_prediction_with_their_ratings.png'
    plt.savefig(plot_file_path)

    plt.show()
