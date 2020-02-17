from numpy import argmin, array, Inf, sqrt
from numpy.random import permutation

class Body:
    def sample(self, N):
        raise NotImplementedError

    def distances(self, xs, x):
        raise NotImplementedError

    def centroid(self, xs):
        raise NotImplementedError

    def label(self):
        raise NotImplementedError

    def voronoi_iteration(self, N, k, tol):
        samples = self.sample(N)

        def centers_to_clusters(centers):
            cluster_idxs = argmin([self.distances(samples, center) for center in centers], axis=0)
            clusters = [samples[cluster_idxs == idx] for idx in range(k)]
            return clusters

        centers = samples[permutation(N)[:k]]
        clusters = centers_to_clusters(centers)

        def total_distance(clusters, centers):
            return sqrt(sum(sum(self.distances(cluster, center) ** 2) for cluster, center in zip(clusters, centers)))

        prev_dist = Inf
        dist = total_distance(centers, clusters)
        while prev_dist - dist > tol:
            prev_dist = dist
            centers = array([self.centroid(cluster) for cluster in clusters])
            clusters = centers_to_clusters(centers)
            dist = total_distance(centers, clusters)

        return clusters, centers
