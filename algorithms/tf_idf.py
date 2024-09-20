import math
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from utils.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

cnt_table = {}


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


def generate_tf_idf_values(documents):
    for doc in documents:
        for word in cnt_table.keys():
            cnt_table[word].append(tfidf(word, doc.lower(), documents))
    return cnt_table


def question_7():
    documents = []
    tfidf_values = generate_tf_idf_values(documents)
    tfidf_df = pd.DataFrame(tfidf_values)
    tfidf_df.insert(0, 'Description', documents)
    output_file_path = 'tfidf_values.csv'
    tfidf_df.to_csv(output_file_path, index=False)
    print(f"TF-IDF values have been written to {output_file_path}")


def create_vocab_set(sentences):
    uniq_words = set()
    for sentence in sentences:
        for word in sentence.split(" "):
            word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
            if word.isalnum() and word.lower() not in uniq_words:  # Check if the word contains only alphabetic characters
                uniq_words.add(word.lower())
    return list(uniq_words)


def computing_tfidf_vectors(sentences):
    uniq_words = create_vocab_set(sentences)
    tfidf_vectors = []
    idf_values = [idf(word, sentences) for word in uniq_words]
    for sentence in sentences:
        tfidf_vector = []
        for index, word in enumerate(uniq_words):
            tfidf_value = tf(word, sentence) * idf_values[index]
            tfidf_vector.append(tfidf_value)
        tfidf_vectors.append(tfidf_vector)
    return np.array(tfidf_vectors)


def compute_similarity_matrix(sentences, threshold=0.05):
    tfidf_matrix = computing_tfidf_vectors(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity_matrix[similarity_matrix < threshold] = 0
    return similarity_matrix


def get_top_tfidf_words(summery, num_top_words=3):
    top_words = [["", 0], ["", 0], ["", 0]]
    for doc in summery:
        clean_words_doc = create_vocab_set([doc])
        tfidf_vector = [tfidf(word, doc, summery) for word in clean_words_doc]
        tfidf_dict = {word: tf_idf for word, tf_idf in
                      zip(clean_words_doc, tfidf_vector)}
        temp_top_words = sorted(tfidf_dict, key=tfidf_dict.get, reverse=True)[
                         :num_top_words]
        for top_word in temp_top_words:
            for i, outer_top_word in enumerate(top_words):
                tf_idf_val = tfidf_dict[top_word]
                if tf_idf_val > outer_top_word[1]:
                    top_words[i][0] = top_word
                    top_words[i][1] = tf_idf_val
                    break
    return top_words


def compute_tf_idf_for_top_words_for_each_fruit(df, sentences,
                                                total_top_words):
    uniq_words = create_vocab_set(sentences)
    idf_values = [idf(word, sentences) for word in uniq_words]
    tf_idf_vectors_for_each_fruit_summery = []
    for fruit_summery in sentences:
        curr_fruit_tf_idf_values = []
        for index, word in enumerate(uniq_words):
            curr_word_tf_idf_value = tf(word, fruit_summery) * idf_values[
                index]
            curr_fruit_tf_idf_values.append(
                [curr_word_tf_idf_value, word, index])
        sorted_by_tf_idf_values = sorted(curr_fruit_tf_idf_values,
                                         key=lambda x: x[0], reverse=True)
        best_words = [[elem[1], elem[2]] for elem in
                      sorted_by_tf_idf_values[:3]]
        for top_word_info in best_words:
            total_top_words.add((top_word_info[0], top_word_info[1]))
        tf_idf_vectors_for_each_fruit_summery.append(curr_fruit_tf_idf_values)
    df_top_words = pd.DataFrame(index=[i for i in range(20)],
                                columns=[elem[0] for elem in total_top_words])
    for index, fruit_summery in enumerate(sentences):
        for top_word in total_top_words:
            df_top_words.at[index, top_word[0]] = \
                tf_idf_vectors_for_each_fruit_summery[index][top_word[1]][0]
    merged_df = pd.concat([df, df_top_words], axis=1)
    merged_df.to_csv("top_words_tf_idf_values.csv")


def create_word_cloud(tfidf_vector, feature_names):
    # Create a dictionary of words and their corresponding TF-IDF scores
    tfidf_dict = {word: score for word, score in zip(feature_names, tfidf_vector)}

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_dict)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # No axes for word cloud
    plt.title("Word Cloud of PostFirstSen based on TF-IDF", fontsize=16)
    plt.show()


def analyze_tfidf_impact(df):
    # Calculate TF-IDF for the 'PostFirstSen' column
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['PostFirstSen'])

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Add PostRating to the TF-IDF DataFrame for analysis
    tfidf_df['PostRating'] = df['PostRating']

    # Calculate mean TF-IDF scores for each word
    mean_tfidf = tfidf_df.mean().drop('PostRating')

    # Create a word cloud for the mean TF-IDF values
    create_word_cloud(mean_tfidf.values, mean_tfidf.index)

    # Optional: Analyze correlation between TF-IDF values and PostRating
    correlation = tfidf_df.corr()['PostRating'].drop('PostRating')

    # Plot correlation
    plt.figure(figsize=(14, 7))
    correlation.plot(kind='bar', color='darkblue')
    plt.title("Correlation of TF-IDF Values with PostRating", fontsize=16)
    plt.ylabel("Correlation Coefficient", fontsize=14)
    plt.xticks(rotation=45)
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.show()
