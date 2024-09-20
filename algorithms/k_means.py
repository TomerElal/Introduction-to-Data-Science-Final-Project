import matplotlib.pyplot as plt

from utils.metrics import *


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

        for cluster in clusters:
            if not cluster:
                # empty cluster - init a random centroid to it
                random_point = X.sample(n=1)
                new_centroid_values = random_point.to_numpy().flatten()  # Get the point values
                random_index = random_point.index[0]  # Get the index of the point
                cluster.append(np.append(new_centroid_values, random_index))

        new_centroids = update_centroid_func(clusters)
        for i, new_cen in enumerate(new_centroids):
            if np.any(np.isnan(new_cen)):
                new_centroids[i] = initial_centroids[i]
        # Check for convergence
        if np.all(np.array(centroids) == np.array(new_centroids)):
            break
        centroids = new_centroids

    plot_clusters(full_df, clusters, samples_data.shape[1], title, features, centroids)
    return clusters, centroids


def plot_clusters(df, clusters, num_cols, title, features, centroids=None):
    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    # Scatter plot for each cluster
    colors = ['r', 'g', 'b']  # Add more colors if you have more than 3 clusters

    for i, cluster in enumerate(clusters):
        feature_np_a = np.array(
            [df.at[point[num_cols], features[0]] for point in cluster])
        feature_np_b = np.array(
            [df.at[point[num_cols], features[1]] for point in cluster])

        # Plot the cluster points
        plt.scatter(feature_np_a, feature_np_b, c=colors[i],
                    label=f'Cluster {i + 1}', alpha=0.7)

    # Add labels and title
    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.grid(True)

    # Save the plots
    plot_file_path = f'plots/{title.lower()}.png'
    plt.savefig(plot_file_path)

    plt.show()


def apply(df, k_means_features, plot_features, title, k=2, func=euclidean_distance,
          update_centroid_func=mean_centroid_func):
    feature_vectors = df[k_means_features]
    clusters, centroids = apply_kmeans(full_df=df, samples_data=feature_vectors,
                                       features=plot_features,
                                       k=k, max_iters=100,
                                       func=func,
                                       title=title,
                                       update_centroid_func=update_centroid_func)
