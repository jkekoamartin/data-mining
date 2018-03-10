import sys
import numpy
import itertools
from collections import Counter
import pandas as pd
import math

import timeit

start = timeit.default_timer()


# Code written by James Martin


class Bayes_Classifier:

    def __init__(self, train_set, test_set, output):
        # un-partitioned training data
        self.train_set = train_set
        # test data
        self.test_set = test_set
        # output file
        self.output = output

        # attributes are df[1:] attributes mapped to set of unique attribute columns
        self.attributes = {}

        # contains class labels in a list
        self.classifiers = []

        self.test_results = []
        self.accuracy = 0.0

        # note that test and training data contains class labels for accuracy testing
        self.train_df = pd.read_csv(self.train_set, header=None, delimiter=r"\s+")
        self.test_df = pd.read_csv(self.test_set, header=None, delimiter=r"\s+")

    def preprocess(self):
        # get attributes from training data, assign label
        # get length of row - 1
        # give name to each attribute [a1,a2,...,a<n>]
        # assign to attributes
        # print(len(self.train[:]))
        train_df = self.train_df
        attributes = self.attributes

        # for each column of train, get unique attribute values and store in dict with index as the key
        for column in train_df:
            unique_values = set(train_df[column])
            attributes[column] = unique_values

        # pop 0 from dict, which contains classifiers
        self.classifiers = list(attributes.pop(0))

    def learn(self):
        attributes_dict = self.attributes

        # get class probabilities
        class_probabilities = {}

        # counts of e and p
        class_count_dict = dict(self.train_df[0].value_counts())

        # total size of data
        total_size = sum(class_count_dict.values())

        for c_label in class_count_dict:
            class_probabilities[c_label] = class_count_dict[c_label] / total_size

        data = []

        partitions = get_partitions(attributes_dict, self.train_df)

        test_data = self.test_df

        nump_arr = test_data.as_matrix()

        for line in nump_arr:
            prediction = classify_tuple(line, partitions, class_probabilities, total_size)
            line = list(line)
            line.append(prediction)
            data.append(line)

        self.test_results = data

    def classify(self):
        pass

    def get_accuracy(self):

        correct = 0.0
        classified = len(self.test_results)

        for line in self.test_results:
            if str(line[0]) == str(line[-1]):
                correct += 1
            else:
                continue
        accuracy = (correct / classified) * 100
        print(accuracy)
        self.accuracy = accuracy

    def write(self):

        out = self.output

        out = open(out, 'w')

        output = self.test_results

        for line in output:
            out.write(str(line[1:-1]) + " is classified as: " + str(line[-1]) + ". True class: " + str(line[0]) + "\n")

        out.write("The accuracy is: " + str(self.accuracy) + "%")

        out.close()


def classify_tuple(line, partitions, class_probabilities, total_size):
    # test tuple = [x,w,q,t,c,s,a]
    # test partition unpacking

    e_prob = 0
    p_prob = 0

    class_count_dict = dict(self.train_df[0].value_counts())

    # skip first index -- it contains the classifier
    attribute = 1
    for column in line[1:]:
        pass

    for part in partitions:
        attribute_partition = partitions[part]
        # for each value in the attribute
        for value in attribute_partition:
            value_partition = pd.DataFrame(attribute_partition[value])
            # this is the total count of the attribute in it's partition
            print(int(value_partition[part].value_counts()))

    return "p"


def get_partitions(attribute_dict, partition):
    # this return the partitions from splitting on atrribute values, used to calc info gain

    partitions = {}
    # to do row operations on partition
    nump_array = partition.as_matrix()

    # for each attribute column
    for attribute in attribute_dict:
        # dict for entire column
        attribute_partition = {}
        attribute_value = attribute_dict[attribute]
        # for each value in attribute column
        for value in attribute_value:
            # make a partition of just the attribute values
            temp_partition = []
            for row in nump_array:
                if str(row[attribute]) == str(value):
                    temp_partition.append(row)
            # this maps a attribute value to its partition
            attribute_partition[value] = temp_partition
        # this maps an attribute column to all of its sub partitions
        partitions[attribute] = attribute_partition
    # now calculate a probability for each partition?

    return partitions


def run():
    # get args
    training, testing, output = sys.argv[1:]

    bayes = Bayes_Classifier(training, testing, output)

    bayes.preprocess()
    bayes.learn()
    bayes.get_accuracy()
    bayes.write()
    print("Complete. Results written to " + "'" + output + "'")


def test(train_set, test_set, out):
    bayes = Bayes_Classifier(train_set, test_set, out)

    bayes.preprocess()
    bayes.learn()
    bayes.get_accuracy()
    bayes.write()

    print("Complete. Results written to " + "'" + out + "'")


if __name__ == "__main__":
    # check correct length args
    if len(sys.argv) == 1:
        print("No arguments passed, running test mode. Test args: [mushroom.training training.test outputc45.dat]")
        test("mushroom.training", "mushroom.test", "outputBayes.dat")
    elif len(sys.argv[1:]) == 3:
        print("Generating results")
        run()
    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")

stop = timeit.default_timer()

print("Results in " + str(stop - start) + " seconds")
