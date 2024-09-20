import os
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ExtractInfoFromGPT import extract_info_from_gpt  # Do not remove!
import json
from utils import utils
import pandas as pd
import tiktoken

# Load the configuration from JSON file using UTF-8 encoding
with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
prefix_prompt = config['prefix_prompt']

# Initialize the tokenizer for GPT-4
tokenizer = tiktoken.encoding_for_model("gpt-4o")

# Constants
CONFIG_FILE = 'config.json'
CSV_FILE_NAME = 'Linkedin_Posts.csv'
DATASET_COLUMNS = [
    'UserName', 'NumFollowers', 'ContentFirstLine', 'NumWords', 'NumPunctuation', 'NumEmojis', 'NumHashtags',
    'NumLinks',
    'NumLineBreaks', 'HasImage', 'HasVideo', 'PostMainSubject', 'PostMainFeeling',
    'NumReactions', 'NumComments', 'NumShares'
]
LOGIN_URL = 'https://www.linkedin.com/login'
LINKEDIN_PROFILE_URL_TEMPLATE = 'https://www.linkedin.com/in/{}/'
LINKEDIN_ACTIVITY_URL_TEMPLATE = 'https://www.linkedin.com/in/{}/recent-activity/all/'
USERNAME_FIELD_ID = 'username'
PASSWORD_FIELD_ID = 'password'
LOGIN_BUTTON_XPATH = '//button[@type="submit"]'
POST_CONTAINER_ID = 'fie-impression-container'
POST_CONTAINER_CLASS = "feed-shared-update-v2"
SHOW_MORE_BUTTON_XPATH = ("//button[contains(@class, 'scaffold-finite-scroll__load-button')"
                          " and contains(., 'Show more results')]")
EMOJIS_CONFIG_KEY = 'emojis'
NUM_OF_POSTS_THRESHOLD = 20  # Minimum posts required
NUM_OF_TOKENS_THRESHOLD = 29000
NUM_OF_POSTS_LIMIT = 100  # Maximum posts possible
USERS_CONFIG_KEY = 'users'
MY_USER_EMAIL = "username@gmail.com"  # Replace with your LinkedIn email
MY_USER_PASSWORD = "password"  # Replace with your LinkedIn password

# Load configuration from a JSON file
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    config = json.load(file)


def extract_user_posts_attributes(driver, user_name):
    """
    Extract attributes of user posts.

    Parameters:
    driver (webdriver): The Selenium WebDriver instance.
    user_name (str): The LinkedIn username.

    Returns:
    list: A list of post attributes.
    """

    posts = get_user_posts(driver, user_name)
    posts_attributes = []
    if len(posts) < NUM_OF_POSTS_THRESHOLD:
        return posts_attributes
    total_tokens = 0
    multiplier = 1
    for post in posts:
        try:
            post_content = (post.find('div', class_='feed-shared-update-v2__description-wrapper').
                            find('span', class_='break-words').text.strip())

            if is_current_post_should_be_ignored(post, post_content):
                continue

            # if total_tokens > NUM_OF_TOKENS_THRESHOLD * multiplier:
            #     multiplier += 1
            #     time.sleep(60)

            print(f"\nStarted Gpt response for {user_name}.")
            gpt_response = extract_info_from_gpt(post_content, prefix_prompt)
            curr_num_of_tokens = (len(tokenizer.encode(prefix_prompt + post_content))
                                  + len(tokenizer.encode(gpt_response)))
            print("Curr post tokens used: " + str(curr_num_of_tokens))
            total_tokens += curr_num_of_tokens
            gpt_response = gpt_response.strip().split('\n')
            post_main_subject, post_main_feeling = gpt_response[0], gpt_response[1]
            print(f"Finished Gpt response for {user_name}.")

            # post_main_subject, post_main_feeling = None, None

            comments, has_image, has_video, reactions, shares = extract_data_from_post(post)
            post_attributes = compute_post_attributes(
                comments, post_content, has_image, has_video, reactions,
                shares, post_main_subject, post_main_feeling, post
            )
            posts_attributes.append(post_attributes)
        except Exception as e:
            print(e)
            continue

    print("Total tokens used: " + str(total_tokens))
    return posts_attributes


def is_current_post_should_be_ignored(post, post_content):
    # Check there is an actual post and not just a basic repost.
    if "reposted this" in post.text:
        return True

    # Check the post has enough content.
    if len(post_content.split()) < 5:
        return True

    # Check that the post got enough "baking time" in LinkedIn - so it got all the traffic it deserves.
    time_element = post.find('span', class_='update-components-actor__sub-description')
    time_text = ''
    if time_element:
        time_text = time_element.get_text(strip=True)
    if 'hour' in time_text or 'minutes' in time_text:  # Indicates the post was post less than a day ago.
        return True

    return False


def compute_post_attributes(comments, content, has_image, has_video, reactions,
                            shares, post_subject, post_feeling, post):
    """
    Compute attributes for a single post.

    Parameters:
    comments (str): Number of comments.
    content (str): Post content.
    has_image (bool): Whether the post has an image.
    has_video (bool): Whether the post has a video.
    reactions (str): Number of reactions.
    shares (str): Number of shares.
    post_subject (str): Main subject of the post.
    post_feeling (str): Main feeling of the post.
    post (BeautifulSoup): The BeautifulSoup object of the post.

    Returns:
    list: A list of computed post attributes.
    """
    num_of_words_in_post = utils.count_words(content)
    num_of_punctuation_in_post = utils.count_punctuation(content)
    num_of_hashtags_in_post = utils.count_hashtags(content)
    num_of_links_in_post = utils.count_links(content)
    num_of_line_breaks_in_post = utils.count_line_breaks(post)
    num_of_reactions_in_post = utils.extract_int_from_string(reactions)
    num_of_comments_in_post = utils.extract_int_from_string(comments)
    num_of_shares_in_post = utils.extract_int_from_string(shares)
    post_success_rate = compute_post_success_rate(
        num_of_comments_in_post,
        num_of_reactions_in_post,
        num_of_shares_in_post
    )
    num_of_emojis_in_post = utils.count_emojis(content, config[EMOJIS_CONFIG_KEY])
    first_line = utils.get_content_first_line(content)
    posts_attribute = [
        first_line, num_of_words_in_post, num_of_punctuation_in_post, num_of_emojis_in_post,
        num_of_hashtags_in_post, num_of_links_in_post, num_of_line_breaks_in_post,
        has_image, has_video, post_subject, post_feeling, num_of_reactions_in_post,
        num_of_comments_in_post, num_of_shares_in_post
    ]
    return posts_attribute


def extract_data_from_post(post):
    """
    Extract data from a single post.

    Parameters:
    post (BeautifulSoup): The BeautifulSoup object of the post.

    Returns:
    tuple: A tuple containing comments, has_image, has_video, reactions, and shares.
    """
    try:
        reactions = post.find('button', {'aria-label': lambda x: x and 'reactions' in x}).find('span').text.strip()
    except AttributeError:
        reactions = '0'
    try:
        comments = post.find('button', {'aria-label': lambda x: x and 'comments' in x}).find('span').text.strip()
    except AttributeError:
        comments = '0'
    try:
        shares = post.find('button', {'aria-label': lambda x: x and 'reposts' in x}).find('span').text.strip()
    except AttributeError:
        shares = '0'
    try:
        image_container = post.find('div', class_='update-components-image')
        has_image = image_container is not None and image_container.find('img') is not None
    except AttributeError:
        has_image = False
    try:
        has_video = post.find('video') is not None
    except AttributeError:
        has_video = False
    return comments, has_image, has_video, reactions, shares


def wait_for_posts_to_load(driver, timeout=10):
    """Wait for the LinkedIn posts to load by checking if any post container is present."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CLASS_NAME, POST_CONTAINER_CLASS))
        )
    except Exception:
        pass


def scroll_page(driver):
    """Scroll to the bottom of the page to trigger loading more posts."""
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(2)  # Short pause to allow dynamic content to load


def click_show_more_button(driver):
    """Click the 'Show more' button if it exists."""
    try:
        show_more_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, SHOW_MORE_BUTTON_XPATH))
        )
        if show_more_button:
            driver.execute_script("arguments[0].click();", show_more_button)
            time.sleep(2)  # Short pause to allow more posts to load
    except Exception:
        pass  # If the button is not found, just continue


def get_user_posts(driver, user_name):
    """
    Retrieve user posts from LinkedIn.

    Parameters:
    driver (webdriver): The Selenium WebDriver instance.
    user_name (str): The LinkedIn username.
    scrolls (int): The number of times to scroll to load more posts.

    Returns:
    list: A list of BeautifulSoup objects representing the posts.
    """
    # Navigate to the user's LinkedIn activity page
    driver.get(LINKEDIN_ACTIVITY_URL_TEMPLATE.format(user_name))

    # Wait for the page to load posts initially
    wait_for_posts_to_load(driver)

    previous_post_count = 0
    ordered_posts_set = dict.fromkeys(set())

    while len(ordered_posts_set) < NUM_OF_POSTS_LIMIT:
        # Scroll to the bottom of the page to load more posts
        scroll_page(driver)

        # Click the "Show more" button if it appears
        click_show_more_button(driver)

        # Parse the current page content using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find all posts by their container class
        new_posts = soup.find_all('div', class_=POST_CONTAINER_CLASS)

        # If no new posts are loaded, break the loop
        if len(new_posts) == previous_post_count:
            break
        previous_post_count = len(new_posts)

        ordered_posts_set.update(dict.fromkeys(new_posts))

    print(f"\nTotal number of posts retrieved: {len(ordered_posts_set)}")
    return ordered_posts_set


def compute_post_success_rate(num_of_comments, num_of_reactions, num_of_shares):
    """
    Compute the success rate of a post.

    Parameters:
    num_of_comments (int): Number of comments.
    num_of_reactions (int): Number of reactions.
    num_of_shares (int): Number of shares.

    Returns:
    int: The success rate of the post.
    """
    return num_of_reactions + num_of_comments + num_of_shares


# TODO: Need to fix this function (not used right now)
def extract_detailed_reactions(post):
    """
    Extract detailed reactions from a post.

    Parameters:
    post (BeautifulSoup): The BeautifulSoup object of the post.

    Returns:
    dict: A dictionary with reaction types and counts.
    """
    detailed_reactions = {}
    reaction_buttons = post.find_all('button', {'role': 'tab'})

    for button in reaction_buttons:
        data_tab = button.get('data-js-reaction-tab')
        if data_tab:
            img_tag = button.find('img')
            count_span = button.find_all('span')[1]
            if img_tag and count_span:
                reaction_type = img_tag['alt']
                reaction_count = count_span.text.strip()
                detailed_reactions[reaction_type] = int(reaction_count)

    return detailed_reactions


def extract_user_attributes(driver, user_name):
    """
    Extract attributes of a LinkedIn user.

    Parameters:
    driver (webdriver): The Selenium WebDriver instance.
    user_name (str): The LinkedIn username.

    Returns:
    list: A list containing the username and number of followers.
    """
    driver.get(LINKEDIN_PROFILE_URL_TEMPLATE.format(user_name))
    time.sleep(3)
    try:
        followers = driver.find_element(By.XPATH, '//p[contains(@class, "text-body-small")]/span').text
    except:
        followers = 'N/A'
    num_of_followers = utils.extract_int_from_string(followers)
    return [user_name, num_of_followers]


def login():
    """
    Log in to LinkedIn.

    Returns:
    webdriver: The Selenium WebDriver instance after login.
    """
    driver = webdriver.Chrome()
    driver.get(LOGIN_URL)

    username = driver.find_element(By.ID, USERNAME_FIELD_ID)
    username.send_keys(MY_USER_EMAIL)

    password = driver.find_element(By.ID, PASSWORD_FIELD_ID)
    password.send_keys(MY_USER_PASSWORD)
    driver.find_element(By.XPATH, LOGIN_BUTTON_XPATH).click()

    time.sleep(12)  # Give time for login to complete
    return driver


def add_rows_to_dataset(user_attributes, posts_attributes):
    """
    Add rows to the dataset.

    Parameters:
    df (DataFrame): The DataFrame to add rows to.
    user_attributes (list): The attributes of the user.
    posts_attributes (list): The attributes of the posts.

    Returns:
    DataFrame: The updated DataFrame.
    """
    new_rows = []
    for post_attributes in posts_attributes:
        new_rows.append(user_attributes + post_attributes)
    add_to_csv = pd.DataFrame(data=new_rows)
    add_to_csv.to_csv(CSV_FILE_NAME, mode='a', header=False, index=False, encoding='utf-8')


def extract_all_data(linkedin_users: set, driver):
    """
    Extract all data for the given LinkedIn users.

    Parameters:
    linkedin_users (set): List of LinkedIn usernames.
    driver (webdriver): The Selenium WebDriver instance.
    df (DataFrame): The DataFrame to add the data to.
    """
    for user in linkedin_users:

        try:
            user_attributes = extract_user_attributes(driver, user)
            posts_attributes = extract_user_posts_attributes(driver, user)
            add_rows_to_dataset(user_attributes, posts_attributes)
        except Exception as e:
            print(f"An error occurred with user {user}: {e}")
            continue


def create_csv_file(file_name):
    # Check if the file exists
    if not os.path.exists(file_name):
        # Create a DataFrame with the specified columns and no rows
        df = pd.DataFrame(columns=DATASET_COLUMNS)
        # Save the DataFrame to a CSV file
        df.to_csv(file_name, index=False)
        print(f"File '{file_name}' created with columns: {DATASET_COLUMNS}")
    else:
        print(f"File '{file_name}' already exists.")


if __name__ == '__main__':
    users = set(config[USERS_CONFIG_KEY])
    chrome_driver = login()
    create_csv_file(CSV_FILE_NAME)
    extract_all_data(users, chrome_driver)  # Main Logic here.
    chrome_driver.quit()
    print(f"Check your project directory, you should see csv file named '{CSV_FILE_NAME}' with the data results.")
