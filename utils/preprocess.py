import random

import pandas as pd

# from sklearn.preprocessing import MinMaxScaler
from utils.constants import PostFields
from utils.eval_post_rating import *


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, post_rating_eval_method=engagement_rating):
    # Handle categorical features: HasImage, HasVideo
    df[PostFields.HAS_IMAGE.value] = df[PostFields.HAS_IMAGE.value].astype(int)
    df[PostFields.HAS_VIDEO.value] = df[PostFields.HAS_VIDEO.value].astype(int)

    # Create dummy variables for POST_MAIN_SUBJECT and POST_MAIN_FEELING
    df = pd.get_dummies(df,
                        columns=[PostFields.POST_MAIN_SUBJECT.value, PostFields.POST_MAIN_FEELING.value],
                        drop_first=False,
                        dtype=int)  # Ensure 1/0 values

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if "Has" not in col and "Main" not in col]

    # scaler = MinMaxScaler()
    # df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Update 'NumFollowers' where 'UserName' is 'bar-rhamim'
    df.loc[df['UserName'] == 'bar-rhamim', 'NumFollowers'] = 14954

    # Add PostRating based on the chosen method
    df['PostRating'] = df.apply(post_rating_eval_method, axis=1)

    # Replacing cols order
    cols = df.columns.tolist()
    cols[1], cols[2] = cols[2], cols[1]
    cols[2], cols[3] = cols[3], cols[2]
    df = df[cols]

    # Divide each post to a bucket 1 from 1-100
    normalize_column(df, PostFields.POST_RATING.value)

    normalize_aesthetic_attributes('data_mining/Linkedin_Posts.csv')

    return pd.read_csv('data_mining/Linkedin_Posts.csv')


def assign_virality_group(post_ratings, group_start_points):
    group_numbers = []

    for rating in post_ratings:
        if rating == 0:
            group_numbers.append(0)
            continue
        for i in range(len(group_start_points) - 1):
            if group_start_points[i] <= rating < group_start_points[i + 1]:
                group_numbers.append(i + 1)
                break
        else:
            group_numbers.append(len(group_start_points))

    return pd.Series(group_numbers, index=post_ratings.index)


def normalize_column(df, column_name):
    group_start_points = create_virality_groups(df[column_name], num_groups=100)
    df[column_name] = assign_virality_group(df[column_name], group_start_points)
    df.to_csv('data_mining/Linkedin_Posts.csv', index=False)


def create_virality_groups(post_ratings, num_groups=100):
    filtered_ratings = post_ratings[post_ratings > 0]
    sorted_ratings = filtered_ratings.sort_values().reset_index(drop=True)
    samples_per_group = (len(sorted_ratings) // num_groups) + 1
    group_start_points = []
    for i in range(0, len(sorted_ratings), samples_per_group):
        group_start_points.append(sorted_ratings.iloc[i])

    return group_start_points


def normalize_aesthetic_attributes(csv_file):
    df = pd.read_csv(csv_file)

    def normalize_num_emojis(row):
        if row['NumEmojis'] > 10:
            return 10
        else:
            return row['NumEmojis']

    def normalize_num_linebreaks(row):
        if row['NumLineBreaks'] > 100:
            return 10
        else:
            return row['NumLineBreaks'] / 10  # Try division with one '/'

    def normalize_num_punctuation(row):
        if row['NumPunctuation'] > 100:
            return 10
        else:
            return row['NumPunctuation'] / 10  # Try division with one '/'

    # Apply the functions to the respective columns
    df['NumEmojis'] = df.apply(normalize_num_emojis, axis=1)
    df['NumLineBreaks'] = df.apply(normalize_num_linebreaks, axis=1)
    df['NumPunctuation'] = df.apply(normalize_num_punctuation, axis=1)

    # Save the updated dataset back to a new CSV file
    df.to_csv('data_mining/Linkedin_Posts.csv', index=False)
