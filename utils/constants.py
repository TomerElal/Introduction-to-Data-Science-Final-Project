from enum import Enum


class PostFields(Enum):
    USERNAME = 'UserName'
    NUM_FOLLOWERS = 'NumFollowers'
    NUM_WORDS = 'NumWords'
    NUM_PUNCTUATION = 'NumPunctuation'
    NUM_EMOJIS = 'NumEmojis'
    NUM_HASHTAGS = 'NumHashtags'
    NUM_LINKS = 'NumLinks'
    NUM_LINE_BREAKS = 'NumLineBreaks'
    HAS_IMAGE = 'HasImage'
    HAS_VIDEO = 'HasVideo'
    POST_MAIN_SUBJECT = 'PostMainSubject'
    POST_MAIN_FEELING = 'PostMainFeeling'
    NUM_REACTIONS = 'NumReactions'
    NUM_COMMENTS = 'NumComments'
    NUM_SHARES = 'NumShares'
    POST_RATING = 'PostRating'
    CONTENT_FIRST_LINE = 'ContentFirstLine'
    POST_CONTENT = 'PostStart'

    NUMERIC_COLS = [
        NUM_WORDS, NUM_FOLLOWERS,
        NUM_PUNCTUATION, NUM_EMOJIS,
        NUM_HASHTAGS, NUM_LINKS,
        NUM_LINE_BREAKS
    ]

    LABEL_COLS = [NUM_REACTIONS, NUM_COMMENTS, NUM_SHARES, POST_RATING]

    CATEGORICAL_COLS = [HAS_IMAGE, HAS_VIDEO, POST_MAIN_SUBJECT, POST_MAIN_FEELING]
