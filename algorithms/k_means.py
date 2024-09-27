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
    np.random.seed(69)
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
    colors = ['b', 'r', 'g', 'c', 'm', 'y']

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


def plot_cluster_avg_postrating_and_numeric(df, clusters, numeric_cols, title, sub_title):
    for cluster_id, points in enumerate(clusters):
        original_indices = [point[-1] for point in points]
        cluster_df = df.loc[original_indices]

        cluster_avg_values = cluster_df[numeric_cols + ['PostRating']].mean()
        plt.figure(figsize=(12, 8))
        sns.barplot(x=cluster_avg_values.index, y=cluster_avg_values.values, palette='viridis')

        plt.title(f'Average Numeric Values for Cluster {cluster_id + 1} - {title.split("-")[-1].lower()}', fontsize=18)
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel('Average Value', fontsize=14)

        plt.ylim(0, cluster_avg_values.max() * 1.2)

        for index, value in enumerate(cluster_avg_values.values):
            plt.text(index, value + 0.03, f'{value:.2f}', ha='center', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_file_path = f'plots/kmeans_plots/cluster_avg_feature_values_cluster_{cluster_id + 1}_{sub_title}.png'
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
          update_centroid_func=mean_centroid_func, type='numeric', sub_title=''):
    feature_vectors = df[k_means_features]
    clusters, centroids = apply_kmeans(samples_data=feature_vectors,
                                       k=k, max_iters=100,
                                       func=func,
                                       update_centroid_func=update_centroid_func)

    for i, cluster in enumerate(clusters):
        print(f"Length of cluster number {i} is {len(cluster)}")

    # Plot the results
    plot_clusters(df, clusters, feature_vectors.shape[1], title, plot_features, centroids)
    if type == 'numeric':
        plot_cluster_avg_postrating_and_numeric(df, clusters, k_means_features, title, sub_title)


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
