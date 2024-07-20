import string
import re
from bs4 import BeautifulSoup
import json


def count_words(text):
    """
    Count the number of words in a given text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The number of words in the text.

    Example:
    >>> count_words("Hello world! #hashtag")
    2
    """
    # Remove hashtags and emojis
    clean_text = re.sub(r'#\S+', '', text)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)

    # Split the text by spaces and other non-word characters
    words = re.split(r'\s+', clean_text.strip())
    res = len(words)
    return res


def count_punctuation(text):
    """
    Count the number of punctuation marks in a given text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The number of punctuation marks in the text.

    Example:
    >>> count_punctuation("Hello, world!")
    2
    """
    res = 0
    for char in text:
        if char in string.punctuation:
            res += 1
    return res


def count_hashtags(text):
    """
    Count the number of hashtags in a given text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The number of hashtags in the text.

    Example:
    >>> count_hashtags("Hello #world! #python")
    2
    """
    res = 0
    for char in text:
        if char == '#':
            res += 1
    return res


def count_links(text):
    """
    Count the number of links in a given text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The number of links in the text.

    Example:
    >>> count_links("Check this out: https://example.com and also https://another.com")
    2
    """
    res = 0
    for word in text.split():
        if 'https://' in word:
            res += 1
    return res


def count_line_breaks(post_html):
    """
    Count the number of line breaks in the given HTML content.

    Parameters:
    post_html (BeautifulSoup): The input HTML content parsed with BeautifulSoup.

    Returns:
    int: The number of line breaks in the HTML content.

    Example:
    >>> html_content = BeautifulSoup('<span class="break-words tvm-parent-container"><br><br></span>', 'html.parser')
    >>> count_line_breaks(html_content)
    2
    """
    return len(post_html.find('span', class_='break-words tvm-parent-container').find_all('br'))


def count_emojis(text, emojis):
    """
    Count the number of emojis in a given text.

    Parameters:
    text (str): The input text.
    emojis (set): A set of emojis to count in the text.

    Returns:
    int: The number of emojis in the text.

    Example:
    >>> count_emojis("Hello world! 😊😊", {'😊'})
    2
    """
    res = 0
    for char in text:
        if char in emojis:
            res += 1
    return res


def extract_int_from_string(s):
    """
    Extract the first integer found in a given string.

    Parameters:
    s (str): The input string.

    Returns:
    int: The first integer found in the string, or 0 if no integer is found.

    Example:
    >>> extract_int_from_string("There are 123 apples.")
    123
    >>> extract_int_from_string("No numbers here.")
    0
    """
    number_str = ''
    for char in s:
        if char.isdigit():
            number_str += char
        elif char == ',' or char == '.':
            continue
        elif number_str:
            break
    if number_str:
        number = int(number_str)
        return number
    return 0


check = """
גאה להיות שייך לקבוצה הזו 🤟🏽
קבוצה קטנה שמצליחה להשפיע ולשבש (באופן חיובי) ארגון עצום.

כבוד גדול לקבל גם הערכה על כך!

אגב,
ראיתם את האתר שלנו?
https://lnkd.in/djKMbCcm

"""
CONFIG_FILE = 'config.json'
EMOJIS_CONFIG_KEY = 'emojis'


def run():
    # Load configuration from a JSON file
    with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
        config = json.load(file)
    print(count_emojis(check, config[EMOJIS_CONFIG_KEY]))
    print(count_links(check))
    print(count_words(check))


if __name__ == '__main__':
    run()
