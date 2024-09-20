# LinkedIn Post Metadata Analysis and Virality contribution

To execute the main script, run the following command in your terminal:

```bash
python main_executer.py
```
## Overview

This project focuses on crawling LinkedIn to extract post metadata and analyze it for insights of the virality of posts and other key features. 
By utilizing sophisticated crawling techniques and advanced algorithms, we aim to find patterns in LinkedIn posts that can provide valuable information for users and researchers.

## Objectives

- **Data Extraction:** Develop a unique and efficient crawling method to gather LinkedIn post metadata.
- **Feature Engineering:** Generate additional features using GPT to capture key aspects of the posts, such as their main subject and emotional tone (paid for that :P).
- **Data Processing:** Normalize numeric features and create dummy variables for categorical features.
- **Evaluation of Post Virality:** The evaluation techniques we used to estimate the post virality.
- **Data Analysis:** Use K-means clustering, TF-IDF, Natural Language Processing and Correlation Analysis techniques to analyze post virality and other characteristics.
- **Visualization:** Present findings through various plots and charts for easy interpretation and conclusions.


## Data Extraction

The project involves crawling LinkedIn posts to extract metadata such as:
- **UserName:** The name of the user who made the post.
- **NumFollowers:** The number of followers the user has.
- **NumWords:** The total number of words in the post.
- **NumPunctuation:** The count of punctuation marks in the post.
- **NumEmojis:** The number of emojis used in the post.
- **NumHashtags:** The total number of hashtags included.
- **NumLinks:** The number of links shared in the post.
- **NumLineBreaks:** The count of line breaks in the post.
- **HasImage:** A boolean indicating if the post contains an image.
- **HasVideo:** A boolean indicating if the post contains a video.
- **PostMainSubject:** The main subject or theme of the post.
- **PostMainFeeling:** The emotional tone conveyed in the post.
- **NumReactions:** The total number of reactions the post received.
- **NumComments:** The number of comments on the post.
- **NumShares:** The count of shares the post received.
- **ContentFirstLine:** The first line of the post content.


The data is extracted into CSV format for further analysis.

## Feature Engineering

Using the GPT API, we generate important features that capture the essence of the posts:
- **PostMainSubject:** The central theme of the post.
- **PostMainFeeling:** The emotional tone conveyed in the post.

## Data Processing

The following preprocessing steps are performed on the extracted data:
1. **Normalization:** Numeric features are scaled for consistency.
2. **Encoding Dummies:** Categorical features are converted to dummy variables.
3. **Text Processing:** The text is cleaned and tokenized

## Evaluation of Post Virality

To assess the virality of posts, we employed 3 different methods:

1. **Engagement Rating:** This function calculates the post rating based on engagement metrics (reactions, comments, and shares) relative to the user's. It applies different weights to each feature:
   ```python
   def engagement_rating(row):
   
       num_reactions_factor = 1
       num_comments_factor = 1
       num_shares_factor = 2

       engagement = (num_reactions_factor * row['NumReactions'] +
                     num_comments_factor * row['NumComments'] +
                     num_shares_factor * row['NumShares'])
       if row['NumFollowers'] > 0:
           return engagement / row['NumFollowers']
       return 0
   ```

2. **Content Quality Rating**: Content based features such as the number of words, punctuation marks, emojis, and hashtags. It operates under the assumption that more engaging content—characterized by a higher number of words, emojis, and hashtags tends to go viral.
   ```python
   def content_quality_rating(row):
   
    word_weight = 0.1
    punctuation_weight = 0.2
    emoji_weight = 0.3
    hashtag_weight = 0.4

    return (word_weight * row['NumWords'] +
            punctuation_weight * row['NumPunctuation'] +
            emoji_weight * row['NumEmojis'] +
            hashtag_weight * row['NumHashtags'])
   ```

2. **Multi-factor Rating**: Combines the engagement score, content quality score and media presence to create a combined rating.
   ```python
   def multifactor_rating(row):
   
    engagement_score = engagement_rating(row)
    content_quality_score = content_quality_rating(row)
    media_bonus = 1.5 if row['HasImage'] or row['HasVideo'] else 1.0

    return (0.4 * engagement_score +
            0.4 * content_quality_score +
            0.2 * media_bonus)
    ```

## Analysis and Visualization

We used several algorithms and techniques we learned throughout the semester, such as:
- **K-means Clustering:** This technique groups posts based on their features to identify patterns. We employed two different distance metrics for the clustering process:
  - **Binary Distance:** Used for categorical features to measure similarity based on the presence or absence of attributes.
  - **Euclidean Distance:** Applied to numeric features to assess the similarity of posts based on their quantitative characteristics.


- **NLP Techniques:**
  - **Word Cloud:** This visualization technique helps to represent the frequency of words in a visually appealing manner, where the size of each word indicates its importance in the dataset. It allows us to quickly identify important words and themes within the posts.
  - **Log-Rank Log-Frequency Analysis:** We conducted a log-log analysis of word occurrences to explore the distribution of word frequencies. This method helps in understanding the relationship between word rank and frequency, providing insights into which words are commonly used in LinkedIn posts versus those that are barely used.
  

- **TF-IDF Analysis:** This method identifies the most significant words and phrases in the posts, allowing us to understand the key relations discussed.


- **Correlation Analysis:** Find for each feature how does he correlate with the Virality value of the post for given evalute method.




Visualizations can be found under the plot directory.
You can also run the project to generate them again as long as you want.

## Authors

- Omer Ferster
- Tomer Elal
- Esther Berestetsky