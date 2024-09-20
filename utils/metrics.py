import numpy as np


def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vector1 = np.linalg.norm(vec1)
    norm_vector2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return 1 - cosine_similarity


def binary_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors must have the same length")
    dist = 0
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]:
            dist += 1
    return dist


def euclidean_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors must have the same length")
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)
