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
- **Data Processing:** Preprocess the dataset including Normalizations, Creation of dummies variables for categorical features and more...
- **Evaluation of Post Virality:** The evaluation technique we used to estimate the post virality.
- **Data Analysis:** Used K-means clustering, TF-IDF, Natural Language Processing and Correlation Analysis techniques to analyze post virality and other characteristics.
- **Visualization:** Present findings through various plots and charts for easy interpretation and conclusions.


## Data Extraction

The project involves crawling LinkedIn posts by parsing HTML tags to extract metadata such as:
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
- **PostContent:** The whole post content.


The data is extracted into CSV format for further analysis.

## Feature Engineering

Using the GPT API, we generate important features that capture the essence of the posts:
- **PostMainSubject:** The central theme of the post.
- **PostMainFeeling:** The emotional tone conveyed in the post.

## Data Processing

The following preprocessing steps are performed on the extracted data:
1. **Encoding Dummies:** Categorical features are converted to dummy variables such as 'MainSubject', 'MainFeeling', 'HasImage' and 'HasVideo'.
2. **Text Processing:** The text is cleaned and tokenized
3. **PostRating generate column:** generated with the evaluation method that will be mention below a new column for the PostRating value for each post. 
We then processed each row to a corresponding bucket of score between 100 buckets.

## Evaluation of Post Virality

To assess the virality of posts, we employed the following methods:

**Engagement Rating:** This function calculates the post rating based on engagement metrics (reactions, comments, and shares) relative to the user's. It applies different weights to each feature:
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

## Analysis and Visualization

We used several algorithms and techniques we learned throughout the semester, such as:
- **K-means Clustering:** This technique groups posts based on their features to identify patterns. We employed two different distance metrics for the clustering process:
  - **Binary Distance:** Used for categorical features to measure similarity based on the presence or absence of attributes.
  - **Euclidean Distance:** Applied to numeric features to assess the similarity of posts based on their quantitative characteristics.


- **NLP Techniques:**
  - **Word Cloud:** This visualization technique helps to represent the frequency of words in a visually appealing manner, where the size of each word indicates its importance in the dataset. It allows us to quickly identify important words and themes within the posts.
  - **Log-Rank Log-Frequency Analysis:** We conducted a log-log analysis of word occurrences to explore the distribution of word frequencies. This method helps in understanding the relationship between word rank and frequency, providing insights into which words are commonly used in LinkedIn posts versus those that are barely used.
  

- **TF-IDF Analysis:** We use this method to predict for a given post the top similarities posts that mostly similar to the given post. the similarities are computed with Cosine distance metric from each pair of posts and their corresponding TF-IDF values conducted from a unique set of words from all documents we crawled.
Then after finding the top similar posts, we compute their PostRating average as a prediction to how well the given post will do in the real LinkedIn platform.
- **Baseline Analysis:** We use a baseline metric to compare the results for our prediction to another baseline prediction.
The baseline metric is norm2 distance between normalized samples, we compute for the given post the same flow mentioned as the TF-IDF model flow,
to find the top doc similarities and their average PostRating values.


- **Correlation Analysis:** Find for each feature how does he correlate with the Virality value conducted from all posts together.




Visualizations can be found under the plot directory.
You can also run the project to generate them again as long as you want.

## Authors

- Omer Ferster
- Tomer Elal
- Esther Berestetsky