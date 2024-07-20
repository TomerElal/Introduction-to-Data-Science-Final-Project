import pandas as pd
import matplotlib.pyplot as plt


def visualization_1():
    """
    This function visualizes the average post rating by follower range.
    """
    # Load the dataset
    df = pd.read_csv('Linkedin_Posts_with_Rating.csv')

    # Ensure the columns are of the correct type
    df['NumFollowers'] = pd.to_numeric(df['NumFollowers'], errors='coerce').fillna(0)
    df['PostRating'] = pd.to_numeric(df['PostRating'], errors='coerce').fillna(0)

    # Define the follower ranges
    ranges = [(0, 1000), (1000, 3000), (3000, 11000), (11000, 20000), (20000, 50000)]
    range_labels = ['0-1k', '1k-3k', '3k-11k', '11k-20k', '20k-50k']

    # Calculate the average post rating for each range and count the number of posts in each range
    average_ratings = []
    post_counts = []
    for r in ranges:
        range_df = df[(df['NumFollowers'] >= r[0]) & (df['NumFollowers'] < r[1])]
        average_rating = range_df['PostRating'].mean()
        average_ratings.append(average_rating)
        post_counts.append(len(range_df))

    # Print the number of posts in each range
    for label, count in zip(range_labels, post_counts):
        print(f"Number of posts in {label} range: {count}")

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range_labels, average_ratings, color='skyblue')
    plt.title('Average Post Rating by Follower Range')
    plt.xlabel('Follower Range')
    plt.ylabel('Average Post Rating')
    plt.grid(axis='y')
    plt.savefig('average_post_rating_by_follower_range.png')  # Save the figure
    plt.close()  # Close the plot to avoid displaying it


def visualization_2():
    """
    This function visualizes the average number of likes by word range in a post.
    """
    df = pd.read_csv('Linkedin_Posts_with_Rating.csv')

    # Ensure the necessary columns are of the correct type
    df['NumWords'] = pd.to_numeric(df['NumWords'], errors='coerce').fillna(0)
    df['NumReactions'] = pd.to_numeric(df['NumReactions'], errors='coerce').fillna(0)

    # Filter out posts with more than 300 words and more than 1000 reactions
    df = df[(df['NumWords'] <= 500) & (df['NumReactions'] <= 1000)]

    # Define the word ranges
    range_labels = ['0-20', '20-50', '50-100', '100-150', '150-200', '200-300', '300-400', '400-500']

    # Create a new column for the word range
    df['WordRange'] = pd.cut(df['NumWords'], bins=[0, 20, 50, 100, 150, 200, 300, 400, 500],
                             labels=range_labels, right=False,include_lowest=True)

    # Calculate the average reactions for each range
    avg_reactions_per_range = df.groupby('WordRange', observed=False)['NumReactions'].mean().reindex(range_labels)

    # Calculate the number of posts for each range
    post_counts_per_range = df['WordRange'].value_counts().reindex(range_labels)

    # Print the number of posts for each range
    print("Number of posts in each word range:")
    print(post_counts_per_range)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range_labels, avg_reactions_per_range, marker='o', color='skyblue', linewidth=2)
    plt.title('Average Number of Likes by Word Range')
    plt.xlabel('Word Range')
    plt.ylabel('Average Number of Likes (NumReactions)')
    plt.grid(True)
    plt.savefig('average_likes_by_word_range.png')  # Save the figure
    plt.close()  # Close the plot to avoid displaying it


if __name__ == '__main__':
    visualization_1()
    visualization_2()
