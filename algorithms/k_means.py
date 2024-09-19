import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.constants import PostFields
from sklearn.decomposition import PCA
from utils.preprocess import *


def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans


# Evaluate clustering using silhouette score
def evaluate_clustering(data, clusters):
    score = silhouette_score(data, clusters)
    print(f"Silhouette Score: {score}")
    return score


# Visualize clusters in 2D using PCA
def visualize_clusters(data, clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis')
    plt.title('K-means Clusters (PCA Reduced to 2D)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.show()


if __name__ == "__main__":
    # Load and preprocess the data
    file_path = '../Linkedin_Posts.csv'  # Replace with your actual CSV file
    df = load_data(file_path)
    processed_data = preprocess_data(df)

    # Apply K-means clustering
    clusters, kmeans_model = apply_kmeans(processed_data, n_clusters=3)

    # Add clusters to original DataFrame for further analysis
    df['Cluster'] = clusters

    # Evaluate the clustering
    evaluate_clustering(processed_data, clusters)

    # Visualize the clusters
    visualize_clusters(processed_data, clusters)
