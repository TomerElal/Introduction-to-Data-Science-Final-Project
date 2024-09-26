from algorithms.correlation import *
from utils.constants import PostFields
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


def apply_kmeans(samples_data, k=3, max_iters=100,
                 func=euclidean_distance,
                 update_centroid_func=mean_centroid_func):
    X = samples_data
    # Initialize centroids randomly
    np.random.seed(67)
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

            clusters[closest_centroid_index].append(np.append(point, index))

        for cluster in clusters:
            if not cluster:
                # empty cluster - init a random centroid to it
                random_point = X.sample(n=1)
                new_centroid_values = random_point.to_numpy().flatten()
                random_index = random_point.index[0]
                cluster.append(np.append(new_centroid_values, random_index))

        new_centroids = update_centroid_func(clusters)
        for i, new_cen in enumerate(new_centroids):
            if np.any(np.isnan(new_cen)):
                new_centroids[i] = initial_centroids[i]
        # Check for convergence
        if np.all(np.array(centroids) == np.array(new_centroids)):
            break
        centroids = new_centroids

    return clusters, centroids


def plot_clusters(df, clusters, num_cols, title, features, centroids=None):
    plt.figure(figsize=(10, 6))
    colors = ['m', 'c', 'g', 'r']

    all_feature_np_a = []
    all_feature_np_b = []

    for i, cluster in enumerate(clusters):
        feature_np_a = np.array([df.at[point[num_cols], features[0]] for point in cluster])
        feature_np_b = np.array([df.at[point[num_cols], features[1]] for point in cluster])

        all_feature_np_a.extend(feature_np_a)
        all_feature_np_b.extend(feature_np_b)

        plt.scatter(feature_np_a, feature_np_b, c=colors[i], label=f'Cluster {i + 1}', alpha=0.7)

    # Calculate limits based on percentiles to avoid outliers affecting scale
    x_min, x_max = np.percentile(all_feature_np_a, [1, 99])
    y_min, y_max = np.percentile(all_feature_np_b, [1, 99])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.grid(True)

    plot_file_path = f'plots/kmeans_plots/{title.lower()}.png'
    plt.savefig(plot_file_path)

    plt.show()


def plot_cluster_avg_postrating(clusters, df, title):
    cluster_avg_postrating = {}

    # Calculate the average PostRating for each cluster based on the original indices in df
    for cluster_id, points in enumerate(clusters):
        original_indices = [point[-1] for point in points]
        post_ratings = df.loc[original_indices, 'PostRating']
        cluster_avg_postrating[cluster_id] = np.mean(post_ratings)

    cluster_indices = sorted(cluster_avg_postrating.keys())
    avg_ratings = [cluster_avg_postrating[idx] for idx in cluster_indices]

    plt.figure(figsize=(10, 8))
    cluster_indices = [ind + 1 for ind in cluster_indices]
    sns.barplot(x=cluster_indices, y=avg_ratings, palette='viridis')

    plt.title(f'Average PostRating for each cluster -{title.split("-")[-1].lower()}', fontsize=18)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Average PostRating', fontsize=14)

    plt.ylim(0, max(avg_ratings) * 1.2)
    for index, value in enumerate(avg_ratings):
        plt.text(index, value + 0.03, f'{value:.2f}', ha='center', fontsize=22)

    plt.tight_layout()
    plot_file_path = f'plots/kmeans_plots/cluster_to_avg_rating_plot - {title.split("-")[-1].lower().strip().lstrip()}.png'
    plt.savefig(plot_file_path)

    plt.show()


def plot_clusters_3d(df, clusters, num_cols, title, features, centroids=None):
    # Define colors for clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for each cluster
    for i, cluster in enumerate(clusters):
        feature_np_a = np.array([df.at[point[num_cols], features[0]] for point in cluster])
        feature_np_b = np.array([df.at[point[num_cols], features[1]] for point in cluster])
        feature_np_c = np.array([df.at[point[num_cols], features[2]] for point in cluster])

        ax.scatter(feature_np_a, feature_np_b, feature_np_c, c=colors[i], label=f'Cluster {i + 1}', alpha=0.7)

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.set_title(title)

    ax.legend()
    plot_file_path = f'plots/kmeans_plots/{title.lower().replace(" ", "_")}.png'
    plt.savefig(plot_file_path)

    plt.show()


def apply(df, k_means_features, plot_features, title, k=2, func=euclidean_distance,
          update_centroid_func=mean_centroid_func, type='numeric'):
    feature_vectors = df[k_means_features]
    clusters, centroids = apply_kmeans(samples_data=feature_vectors,
                                       k=k, max_iters=100,
                                       func=func,
                                       update_centroid_func=update_centroid_func)

    # Plot the results
    plot_clusters(df, clusters, feature_vectors.shape[1], title, plot_features, centroids)
    plot_cluster_avg_postrating(clusters, df, title)
    if type == 'numeric':
        plot_kmeans_numeric_correlations(df, clusters, feature_vectors)
    elif type == 'categorical':
        features_group = split_to_logical_categorical_features(df)
        for feature_group in features_group:
            plot_kmeans_categorical_correlations(df, clusters, feature_group, feature_group[-1])


def split_to_logical_categorical_features(df):
    groups = [[PostFields.HAS_VIDEO.value, PostFields.HAS_IMAGE.value, "image_video"]]
    main_subject_group = []
    main_feeling_group = []
    for col in df.columns:
        if col.startswith(PostFields.POST_MAIN_SUBJECT.value):
            main_subject_group.append(col)
        elif col.startswith(PostFields.POST_MAIN_FEELING.value):
            main_feeling_group.append(col)
    main_subject_group.append("main_subject")
    main_feeling_group.append("main_feeling")
    groups.append(main_subject_group)
    groups.append(main_feeling_group)
    return groups
