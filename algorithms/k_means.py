import matplotlib.pyplot as plt
import numpy as np

from utils.preprocess import *


def binary_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors must have the same length")
    dist = 0
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]:
            dist += 1
    return dist


def bit_count_centroid_func(clusters):
    new_centroids = []
    for cluster in clusters:
        if not cluster:
            continue
        # cluster = list of points
        count_dict = {}
        new_cen = []
        for i in range(len(cluster[0])):
            for point in cluster:
                if point[i] in count_dict.keys():
                    count_dict[point[i]] += 1
                else:
                    count_dict[point[i]] = 1
            max_val = max([val for val in count_dict.values()])
            for k, v in count_dict.items():
                if count_dict[k] == max_val:
                    new_cen.append(k)
                    break
        new_centroids.append(new_cen)
    return [cen[:-1] for cen in new_centroids]


def euclidean_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors must have the same length")
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)


def mean_centroid_func(clusters):
    return [np.mean(np.array([elem[:-1] for elem in cluster]), axis=0) for
            cluster in clusters]


def apply_kmeans(full_df, samples_data, features, k=3, max_iters=100, func=euclidean_distance,
                 title='K-means Clustering',
                 update_centroid_func=mean_centroid_func):
    X = samples_data
    # Initialize centroids randomly
    np.random.seed(48)
    initial_indexes = np.random.choice(X.shape[0], k, replace=False)
    initial_centroids = X.iloc[initial_indexes].values.tolist()
    centroids = initial_centroids
    clusters = []
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for index, point in enumerate(X.values.tolist()):
            min_dist = np.inf
            closest_centroid_index = None
            for i, centroid in enumerate(centroids):
                dist_to_centroid = func(point, centroid)
                if dist_to_centroid < min_dist:
                    min_dist = dist_to_centroid
                    closest_centroid_index = i

            # Assuming point is a NumPy array and index is a scalar value
            clusters[closest_centroid_index].append(np.append(point, index))

        new_centroids = update_centroid_func(clusters)
        for i, new_cen in enumerate(new_centroids):
            if np.any(np.isnan(new_cen)):
                new_centroids[i] = initial_centroids[i]
        # Check for convergence
        if np.all(np.array([cen[:-1] for cen in centroids]) == np.array(new_centroids)):
            break
        centroids = new_centroids

    plot_clusters(full_df, clusters, samples_data.shape[1], title, features)
    return clusters, centroids


def plot_clusters(df, clusters, num_cols, title, features):
    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    # Scatter plot for each cluster
    colors = ['r', 'g', 'b']
    for i, cluster in enumerate(clusters):
        feature_np_a = np.array(
            [df.at[point[num_cols], features[0]] for point in cluster])
        feature_np_b = np.array(
            [df.at[point[num_cols], features[1]] for point in cluster])

        plt.scatter(feature_np_a, feature_np_b, c=colors[i],
                    label=f'Cluster {i + 1}', alpha=0.7)

    # Add labels and title
    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.grid(True)
    plt.show()


def first_figure():
    # Apply 1st K-means clustering - Numeric features
    k_means_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    feature_vectors = df[k_means_features]
    clusters, centroids = apply_kmeans(full_df=processed_data, samples_data=feature_vectors,
                                       features=plot_features,
                                       k=3, max_iters=100,
                                       func=euclidean_distance,
                                       title='K-means Clustering - NumReactions, NumComments',
                                       update_centroid_func=mean_centroid_func)


def second_figure():
    # Apply 2nd K-means clustering - Categorical features
    k_means_features = [PostFields.HAS_IMAGE.value, PostFields.HAS_VIDEO.value]
    plot_features = [PostFields.NUM_REACTIONS.value, PostFields.NUM_COMMENTS.value]
    feature_vectors = df[k_means_features]
    clusters, centroids = apply_kmeans(full_df=processed_data, samples_data=feature_vectors,
                                       features=plot_features,
                                       k=2, max_iters=100,
                                       func=binary_distance,
                                       title='K-means Clustering - HasImage, HasVideo',
                                       update_centroid_func=bit_count_centroid_func)


if __name__ == "__main__":
    file_path = '../Linkedin_Posts.csv'
    df = load_data(file_path)
    processed_data = preprocess_data(df)

    first_figure()
    second_figure()
