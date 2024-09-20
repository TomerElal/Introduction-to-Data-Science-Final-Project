import math

from matplotlib import pyplot as plt

from utils.utils import *


def create_vocab_set(sentences):
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


def generate_tf_idf_values(documents, cnt_table):
    for doc in documents:
        for word in cnt_table.keys():
            cnt_table[word].append(tfidf(word, doc.lower(), documents))
    return cnt_table


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
