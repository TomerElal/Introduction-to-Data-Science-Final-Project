import string
from bs4 import BeautifulSoup
import requests
import regex as re
import unicodedata
import emoji


def call_project_translate(data):
    url = "http://127.0.0.1:5000/translate"  # URL of Project Translate API
    response = requests.post(url, json={'data': data})

    if response.status_code == 200:
        result = response.json()['result']
        return result
    else:
        print(f"Failed to communicate with Project B: {response.status_code}")


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
    try:
        return len(post_html.find('span', class_='break-words tvm-parent-container').find_all('br'))
    except Exception as e:
        print(e)
        return 0


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


def get_content_first_line(content):
    """
    Get the first line of content. The first line is defined by the end of a sentence
    ('.', '?', '!', or newline), or a limit of 20 words, whichever comes first.

    Parameters:
    content (str): The input text.

    Returns:
    str: The first line of content.
    """
    # Check for sentence-ending punctuation or newline
    sentence_end_match = re.search(r'([.!?])\s|(\n)', content)
    words = content.split()

    # If we find a sentence-ending punctuation or newline
    if sentence_end_match:
        first_sentence = content[:sentence_end_match.end()].strip()
    elif len(words) > 20:
        first_sentence = ' '.join(words[:20])
    else:
        first_sentence = content.strip()

    # Replace everything that is not alphabetic or punctuation with a space
    first_sentence_cleaned = re.sub(r'[^\p{L}\s.,!?]', ' ', first_sentence)

    translated_sentence = call_project_translate(first_sentence_cleaned)

    # Normalize the string to remove special characters
    translated_sentence = unicodedata.normalize('NFKD', translated_sentence).encode('ascii', 'ignore').decode(
        'utf-8')

    # Remove emojis
    translated_sentence = emoji.replace_emoji(translated_sentence, replace="")

    return translated_sentence


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
CONFIG_FILE = '../config.json'
EMOJIS_CONFIG_KEY = 'emojis'


def run():
    print(get_content_first_line(
        "We are looking for an experienced 𝗗𝗕𝗔 with strong knowledge at 𝗦𝗤𝗟 𝗦𝗲𝗿𝘃𝗲𝗿 𝗗𝗮𝘁𝗮𝗯𝗮𝘀𝗲 to be 𝗼𝗻𝗲 of us!𝗤𝘂𝗮𝗹𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻𝘀:👉3 years working with production environment (Enterprise companies – Huge Advantage).👉 Designing high availability and disaster recovery solutions.👉Strong knowledge of SQL Server features (replication, SSIS, SSRS)👉Experience with Mirroring \ Always on"))


if __name__ == '__main__':
    run()
