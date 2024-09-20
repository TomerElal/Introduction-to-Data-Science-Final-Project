import math
import re

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


def generate_tf_idf_values(documents):
    for doc in documents:
        for word in cnt_table.keys():
            cnt_table[word].append(tfidf(word, doc.lower(), documents))
    return cnt_table


if __name__ == "__main__":
    documents = extract_documents()
    vocab_list = create_vocab_set(documents)
    cnt_table = {word: [] for word in vocab_list}
    generate_tf_idf_values(documents)
    print(cnt_table)
    print(len(cnt_table['telegram']))

