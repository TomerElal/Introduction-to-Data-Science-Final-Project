
def engagement_rating(row):
    """
    This function calculates the post rating based on the engagement metrics
    (reactions, comments, and shares) relative to the user's followers.
    """

    num_reactions_factor = 1
    num_comments_factor = 5
    num_shares_factor = 10

    engagement = (num_reactions_factor * row['NumReactions'] +
                  num_comments_factor * row['NumComments'] +
                  num_shares_factor * row['NumShares'])
    if row['NumFollowers'] > 0:  # Avoid division by zero
        return 100 * engagement / row['NumFollowers']
    return 0
