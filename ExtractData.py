import os
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from ExtractInfoFromGPT import extract_info_from_gpt  # Do not remove!
import json
import utlis
import pandas as pd

# Constants
CONFIG_FILE = 'config.json'
CSV_FILE_NAME = 'Linkedin_Posts.csv'
DATASET_COLUMNS = [
    'UserName', 'NumFollowers', 'NumWords', 'NumPunctuation', 'NumEmojis', 'NumHashtags', 'NumLinks',
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
SHOW_MORE_BUTTON_XPATH = ("//button[contains(@class, 'scaffold-finite-scroll__load-button')"
                          " and contains(., 'Show more results')]")
EMOJIS_CONFIG_KEY = 'emojis'
USERS_CONFIG_KEY = 'users'
MY_USER_EMAIL = "userName@gmail.com"  # Replace with your LinkedIn email
MY_USER_PASSWORD = "Password"  # Replace with your LinkedIn password

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
    posts_contents = []
    gpt_response = extract_gpt_response(posts, posts_contents)
    gpt_response_curr_index = 0
    for i in range(len(posts)):
        should_pass = False
        gpt_response_curr_index, should_pass = is_current_post_should_be_ignored(gpt_response_curr_index, i, posts,
                                                                                 posts_contents, should_pass)
        if should_pass:
            continue

        comments, has_image, has_video, reactions, shares = extract_data_from_post(posts[i])
        post_main_subject = gpt_response[gpt_response_curr_index]
        post_main_feeling = gpt_response[gpt_response_curr_index + 1]
        posts_attribute = compute_post_attributes(
            comments, posts_contents[(gpt_response_curr_index // 2)], has_image,
            has_video, reactions, shares, post_main_subject, post_main_feeling,
            posts[i]
        )
        gpt_response_curr_index += 2
        posts_attributes.append(posts_attribute)

    return posts_attributes


def is_current_post_should_be_ignored(gpt_response_curr_index, i, posts, posts_contents, should_pass):
    # Check there is an actual post and not just a basic repost.
    if "reposted this" in posts[i].text:
        should_pass = True
        return gpt_response_curr_index, should_pass

    # Check the post has enough content.
    if len(posts_contents[(gpt_response_curr_index // 2)].split()) < 5:
        gpt_response_curr_index += 2
        should_pass = True
        return gpt_response_curr_index, should_pass

    # Check that the post got enough "baking time" in LinkedIn - so it got all the traffic it deserves.
    time_element = posts[i].find('span', class_='update-components-actor__sub-description')
    time_text = ''
    if time_element:
        time_text = time_element.get_text(strip=True)
    if 'hour' in time_text:  # Indicates the post was post less than a day ago.
        should_pass = True
    return gpt_response_curr_index, should_pass


def extract_gpt_response(posts, posts_contents):
    """
    Extract GPT response from posts.

    Parameters:
    posts (list): List of posts.
    posts_contents (list): List to store post contents.

    Returns:
    list: GPT response for the posts.
    """
    for post in posts:
        if "reposted this" in post.text:
            continue
        try:
            content = (post.find('div', class_='feed-shared-update-v2__description-wrapper').
                       find('span', class_='break-words').text.strip())
        except AttributeError:
            content = 'N/A'
        posts_contents.append(content)
    # gpt_response = extract_info_from_gpt(posts_contents).split('\n')
    # The next line replaced the above line in order to not waste gpt credits until we'll actually need gpt response.
    gpt_response = [i for i in range(len(posts_contents) * 2)]
    return gpt_response


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
    num_of_words_in_post = utlis.count_words(content)
    num_of_punctuation_in_post = utlis.count_punctuation(content)
    num_of_hashtags_in_post = utlis.count_hashtags(content)
    num_of_links_in_post = utlis.count_links(content)
    num_of_line_breaks_in_post = utlis.count_line_breaks(post)
    num_of_reactions_in_post = utlis.extract_int_from_string(reactions)
    num_of_comments_in_post = utlis.extract_int_from_string(comments)
    num_of_shares_in_post = utlis.extract_int_from_string(shares)
    post_success_rate = compute_post_success_rate(
        num_of_comments_in_post,
        num_of_reactions_in_post,
        num_of_shares_in_post
    )
    num_of_emojis_in_post = utlis.count_emojis(content, config[EMOJIS_CONFIG_KEY])
    posts_attribute = [
        num_of_words_in_post, num_of_punctuation_in_post, num_of_emojis_in_post,
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


def get_user_posts(driver, user_name, scrolls=8):
    """
    Retrieve user posts from LinkedIn.

    Parameters:
    driver (webdriver): The Selenium WebDriver instance.
    user_name (str): The LinkedIn username.
    scrolls (int): The number of times to scroll to load more posts.

    Returns:
    list: A list of BeautifulSoup objects representing the posts.
    """
    driver.get(LINKEDIN_ACTIVITY_URL_TEMPLATE.format(user_name))
    time.sleep(2)  # Initial wait for the page to load

    for _ in range(scrolls):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)  # Wait for more posts to load

        try:
            show_more_button = driver.find_element(By.XPATH, SHOW_MORE_BUTTON_XPATH)
            if show_more_button:
                driver.execute_script("arguments[0].click();", show_more_button)
                time.sleep(2)  # Wait for more posts to load
        except Exception:
            break

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    posts = soup.find_all('div', id=POST_CONTAINER_ID)  # Use the id instead of class
    return posts


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
    num_of_followers = utlis.extract_int_from_string(followers)
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

    time.sleep(3)  # Give time for login to complete
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
    add_to_csv.to_csv(CSV_FILE_NAME, mode='a', header=False, index=False)


def extract_all_data(linkedin_users, driver):
    """
    Extract all data for the given LinkedIn users.

    Parameters:
    linkedin_users (list): List of LinkedIn usernames.
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


def add_post_rate_to_data():
    # Load the dataset
    df = pd.read_csv('Linkedin_Posts.csv')

    # Ensure the columns are of the correct type
    df['NumReactions'] = pd.to_numeric(df['NumReactions'], errors='coerce').fillna(0)
    df['NumComments'] = pd.to_numeric(df['NumComments'], errors='coerce').fillna(0)
    df['NumShares'] = pd.to_numeric(df['NumShares'], errors='coerce').fillna(0)

    # Calculate the PostRating
    df['PostRating'] = df['NumReactions'] + df['NumComments'] + df['NumShares']

    # Save the updated dataset to a new CSV file
    df.to_csv('Linkedin_Posts_with_Rating.csv', index=False)


if __name__ == '__main__':
    users = config[USERS_CONFIG_KEY]
    chrome_driver = login()
    create_csv_file(CSV_FILE_NAME)
    extract_all_data(users, chrome_driver)  # Main Logic here.
    add_post_rate_to_data()
    chrome_driver.quit()
    print(f"Check your project directory, you should see csv file named '{CSV_FILE_NAME}' with the data results.")
