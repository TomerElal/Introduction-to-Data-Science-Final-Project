import math
import nltk
from matplotlib import pyplot as plt

from nltk import pos_tag
from utils.utils import *
from wordcloud import WordCloud


def tokens_occurrences(tokens=None):
    if not tokens:
        return {}
    tokens_occur_dict = {}
    for token in tokens:
        token = token.lower()
        if token in tokens_occur_dict.keys():
            tokens_occur_dict[token] += 1
        else:
            tokens_occur_dict[token] = 1
    return tokens_occur_dict


def create_log_rank_log_freq_sorted_arr(tokens_occur_dict=None):
    """
    :return: 2D array, each entry is array size 3 in form [word_log_rank, word_log_freq, word]
             The returned array will be sorted by the word_log_rank value increasing way.
    """
    if not tokens_occur_dict:
        return []
    res = []
    converted_dict_to_arr = sorted(
        [[k, v] for k, v in tokens_occur_dict.items()], key=lambda x: x[1],
        reverse=True)
    for i in range(len(converted_dict_to_arr)):
        word = converted_dict_to_arr[i][0]
        word_freq = converted_dict_to_arr[i][1]
        res.append([math.log10(i + 1), math.log10(word_freq), word_freq, word])
    return res


def plot_log_log_graph(log_log_arr, title=None, x_axis_title=None, y_axis_title=None):
    x_axis = [elem[0] for elem in log_log_arr]
    y_axis = [elem[1] for elem in log_log_arr]

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, y_axis, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.grid(True)

    # Save the plots
    plot_file_path = f'plots/log_rank_log_freq_plot.png'
    plt.savefig(plot_file_path)

    plt.show()


def extract_proper_nouns(tokens):
    nltk.download('averaged_perceptron_tagger')
    pos_tags = pos_tag(tokens)
    proper_nouns = [word for word, pos in pos_tags if pos in ['NNP', 'NNPS']]
    return proper_nouns


def create_word_cloud(words):
    text = ' '.join(words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the plots
    plot_file_path = f'plots/word_cloud.png'
    plt.savefig(plot_file_path)

    plt.show()


def create_log_info_and_plot(tokens, add_to_title_string=""):
    tokens_occur_dict = tokens_occurrences(tokens)
    log_rank_log_freq = create_log_rank_log_freq_sorted_arr(tokens_occur_dict)
    plot_log_log_graph(log_rank_log_freq,
                       "Log Rank - Log Freq " + add_to_title_string + " Plot",
                       "Log Rank value", "Log Freq value")
    return log_rank_log_freq
