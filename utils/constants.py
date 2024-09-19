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
