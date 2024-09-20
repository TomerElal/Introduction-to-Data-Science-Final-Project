import numpy as np
import pandas as pd

from utils.constants import PostFields


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    # Handle categorical features: HasImage, HasVideo
    df[PostFields.HAS_IMAGE.value] = df[PostFields.HAS_IMAGE.value].astype(int)
    df[PostFields.HAS_VIDEO.value] = df[PostFields.HAS_VIDEO.value].astype(int)

    # Replace POST_MAIN_SUBJECT and POST_MAIN_FEELING with random characters from ['a', 'b', 'c', 'd']
    random_choices = ['a', 'b', 'c', 'd']
    df[PostFields.POST_MAIN_SUBJECT.value] = np.random.choice(random_choices, size=len(df))
    df[PostFields.POST_MAIN_FEELING.value] = np.random.choice(random_choices, size=len(df))

    # Create dummy variables for POST_MAIN_SUBJECT and POST_MAIN_FEELING
    df = pd.get_dummies(df,
                        columns=[PostFields.POST_MAIN_SUBJECT.value, PostFields.POST_MAIN_FEELING.value],
                        drop_first=False,
                        dtype=int)  # Ensure 1/0 values

    return df
