import pandas as pd

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

    # Add PostRating based on the chosen method
    df['PostRating'] = df.apply(post_rating_eval_method, axis=1)

    # Replacing cols order
    cols = df.columns.tolist()
    cols[1], cols[2] = cols[2], cols[1]
    df = df[cols]

    return df
