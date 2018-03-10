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

        self.normalizers = []

        self.attribute_probs = []

        self.size = 0

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
        self.size = total_size

        for c_label in class_count_dict:
            class_probabilities[c_label] = class_count_dict[c_label] / total_size

        # per class label
        attribute_probs = get_probability(attributes_dict, self.train_df, total_size, self.classifiers)

        # per attribute
        normalizers = get_attribute_dict(attributes_dict, self.train_df, total_size)

        self.attribute_probs = attribute_probs
        self.normalizers = normalizers

    def classify(self):
        test_data = self.test_df

        normalizers = self.normalizers
        attribute_probs = self.attribute_probs

        nump_arr = test_data.as_matrix()

        data = []

        for line in nump_arr:
            prediction = classify_tuple(line, attribute_probs, normalizers, self.classifiers)
            line = list(line)
            line.append(prediction)
            data.append(line)

        self.test_results = data

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


def classify_tuple(line, attribute_probs, normalizers, classifiers):

    best_prob = 0
    best_label = ""
    for label in classifiers:
        normalizer = 1
        label_probability = 1
        label_dict = attribute_probs[label]
        for attribute in label_dict:
            value = line[attribute]
            value_dict = label_dict[attribute]
            norm_dict = normalizers[attribute]
            for each in value_dict:
                label_probability *= value_dict[value]
                normalizer *= norm_dict[value]
        if (label_probability / normalizer) > best_prob:
            best_label = label
        else:
            continue

    return best_label


def probability(value, attribute, train_df, total_size, classifier):
    prob = 0

    nump_array = train_df.as_matrix()

    for row in nump_array:
        if row[attribute] == value and row[0] == classifier:
            prob += 1

    prob = prob / total_size

    return prob


def get_probability(attributes_dict, train_df, total_size, classifiers):
    # dicts of probability split on class labels
    prob_class_split = {}

    for classifier in classifiers:

        partition_dict = {}

        for attribute in attributes_dict:
            attribute_value_dict = {}
            attribute_value = attributes_dict[attribute]
            for value in attribute_value:
                attribute_value_dict[value] = probability(value, attribute, train_df, total_size, classifier)
            #     map attribute value probabilities to attribute column
            partition_dict[attribute] = attribute_value_dict
        prob_class_split[classifier] = partition_dict

    return prob_class_split


def attribute_probability(value, attribute, partition, total_size):
    prob = 0

    nump_array = partition.as_matrix()

    for row in nump_array:
        if row[attribute] == value:
            prob += 1

    prob = prob / total_size

    return prob


def get_attribute_dict(attribute_dict, partition, total_size):
    # this return the partitions from splitting on atrribute values, used to calc info gain

    partition_dict = {}

    for attribute in attribute_dict:
        attribute_value_dict = {}
        attribute_value = attribute_dict[attribute]
        for value in attribute_value:
            attribute_value_dict[value] = attribute_probability(value, attribute, partition, total_size)
        #     map attribute value probabilities to attribute column
        partition_dict[attribute] = attribute_value_dict

    return partition_dict


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
    bayes.classify()
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
