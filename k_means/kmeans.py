import math
import random
import sys
import timeit

import pandas as pd

start = timeit.default_timer()


# Code written by James Martin

class Centroid:

    def __init__(self):
        self.label = None
        self.location = []
        self.cluster = []


class K_Means:

    def __init__(self, train_set, k, output):

        # test data
        self.k = k
        # output file
        self.output = output

        # attributes are df[1:] attributes mapped to set of unique attribute columns
        self.attributes = {}

        # contains class labels in a list
        self.classifiers = []

        # note that test and training data contains class labels for accuracy testing
        # Data must be comma delimited
        self.train_df = pd.read_csv(train_set, header=None, delimiter=r",")

        # centroids for clustering
        self.centroids = []

        # results of clustering
        self.cluster_results = {}

    def pre_process(self):
        # todo: sanitize data. If column contains non_numerical data, drop it. i.e generalize this
        # drop last column from data frame
        self.train_df = self.train_df.iloc[:, :-1]

    def initialize(self):
        """
        Initializes random centroids for k-means algorithm
        """
        k = self.k
        df = self.train_df
        rand_centroids = []

        for label in range(int(k)):
            rand_location = []
            for column in df:
                rand_location.append(random.choice(df[column]))

            # assign new centroid
            rand_centroid = Centroid()
            rand_centroid.label = label
            rand_centroid.location = rand_location

            # add the centroid to list of random centroids
            rand_centroids.append(rand_centroid)

        self.centroids = rand_centroids

    def converge(self):
        """
        Converge on centroids. First, cluster on centroids.
        Then take average of cluster. Finally, assign new centroids

        todo: check this logic
        Repeat until centroids no longer move or SSE reaches minimum
        """

        # use this to assign point
        centroids = self.centroids

        # for each data point, calculates dist to each cluster
        nump_arr = self.train_df.as_matrix()

        k = 0

        not_converged = True

        while not_converged:

            # store current centroid locations
            c_locations = [x.location for x in self.centroids]

            for point in enumerate(nump_arr):
                min_centroid = None
                min_dist = float('inf')
                for centroid in centroids:
                    # print(centroid)
                    dist = euclid_dist(point[1], centroid)
                    # update centroid
                    if dist < min_dist:
                        min_centroid = centroid.label
                        min_dist = dist

                # assign cluster number to row number, to maintain output order
                self.cluster_results[point[0]] = min_centroid
                # assign row to cluster number
                self.centroids[min_centroid].cluster.append(point[1])

            self.update_centroids()

            # get new centroid locations
            new_c_locations = [x.location for x in self.centroids]

            # if old and new centroids are the same, stop algorithm
            if c_locations == new_c_locations:
                not_converged = False

            k += 1

    def update_centroids(self):
        """
        assigns new point to each centroid based on the mean of their respective clusters
        """

        new_centroids = []

        for centroid in enumerate(self.centroids):
            # get object
            new_centroid = centroid[1]
            # new location is the average of the cluster
            new_location = self.get_mean(new_centroid.cluster)
            # set new location
            new_centroid.location = list(new_location)
            # clear previous cluster
            new_centroid.cluster.clear()
            # update cluster
            new_centroids.append(new_centroid)

        self.centroids = new_centroids

    def write(self):

        results = self.cluster_results

        out = self.output

        out = open(out, 'w')

        for row in results:
            result = results[row]
            print(results[row])
            out.write(str(result) + "\n")

        out.close()

    def get_mean(self, cluster):
        """
        Returns a point representing the mean of the input cluster
        :param cluster:
            A list of points
        :return:
            A point
        """
        new_centroid = []

        df = pd.DataFrame(cluster)

        for column in df:
            mean = df[column].mean()
            new_centroid.append(mean)

        # if no data points, then average will be zero, so replace with random value
        if not new_centroid:

            df = self.train_df
            rand_location = []

            for column in df:
                rand_location.append(random.choice(df[column]))

            new_centroid = rand_location

        return new_centroid

    def get_SSE(self):
        """
        returns SSE for the resulting clusters
        :return:
        """
        SSE = 0

        return SSE


# todo: test this, I think this works
def euclid_dist(point, centroid):
    """
    Returns a double representing the Euclidean distance between two point
    :param point:
        a data point
    :param centroid:
        a Centroid object
    :return:
        distance between data point and Centroid object
    """
    dist = 0.0

    # implement euclidean dist
    point_2 = centroid.location

    for p in enumerate(point):
        p_val = p[1]
        index = p[0]
        q_val = point_2[index]

        pq = math.pow((p_val - q_val), 2)
        dist += pq

    dist = math.sqrt(dist)

    return dist


def run():
    # get args
    training, k, output = sys.argv[1:]

    k_means = K_Means(training, k, output)

    k_means.pre_process()
    k_means.initialize()
    k_means.converge()
    k_means.write()

    print("Complete. Results written to " + "'" + output + "'")


def test(dataset, test_set, out):
    k_means = K_Means(dataset, test_set, out)

    k_means.pre_process()
    k_means.initialize()
    k_means.converge()
    k_means.write()

    print("Complete. Results written to " + "'" + out + "'")


if __name__ == "__main__":
    # check correct length args
    if len(sys.argv) == 1:
        print("No arguments passed, running test mode. Test args: [iris.data 3 irisOutput.dat]")
        # test set has three class labels. So using k = 3 to test.
        test("iris.data", "3", "irisOutput.dat")
    elif len(sys.argv[1:]) == 3:
        print("Generating results")
        run()
    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")

stop = timeit.default_timer()

print("Results in " + str(stop - start) + " seconds")