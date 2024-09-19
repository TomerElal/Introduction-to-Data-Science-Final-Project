import pandas as pd

from utils.constants import PostFields


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    # Selecting numeric columns to use in clustering
    numeric_columns = [
        PostFields.NUM_FOLLOWERS.value, PostFields.NUM_WORDS.value,
        PostFields.NUM_PUNCTUATION.value, PostFields.NUM_EMOJIS.value,
        PostFields.NUM_HASHTAGS.value, PostFields.NUM_LINKS.value,
        PostFields.NUM_LINE_BREAKS.value, PostFields.NUM_REACTIONS.value,
        PostFields.NUM_COMMENTS.value, PostFields.NUM_SHARES.value,
        PostFields.POST_RATING.value
    ]

    # Handle categorical features: HasImage, HasVideo
    df[PostFields.HAS_IMAGE.value] = df[PostFields.HAS_IMAGE.value].astype(int)
    df[PostFields.HAS_VIDEO.value] = df[PostFields.HAS_VIDEO.value].astype(int)

    return df
