import sys
import numpy
import itertools
from collections import Counter
import pandas as pd
import math

import timeit

start = timeit.default_timer()


# Code written by James Martin


class DecisionTree:
    def __init__(self, root):
        self.root = root


class C45_Node:
    def __init__(self,
                 parent=None,
                 attribute_name=None,
                 classification=None,
                 attribute_value=None,
                 terminal=False,
                 height=0,
                 partition=None):
        # self.attribute_split_index = attribute_split_index

        self.partition = partition
        self.attribute_name = attribute_name

        self.parent = parent

        # the list of Nodes with attribute values that correspond to the Nodes attribute name
        self.children = []

        self.classification = classification

        # this is the attribute value that link the node to the parent
        self.attribute_value = attribute_value

        self.height = height

        # flag for if terminal, should have a classification if terminal
        self.terminal = terminal

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class C45_Classifier:

    def __init__(self, train_set, test_set, output):
        self.train_set = train_set
        self.test_set = test_set
        self.output = output

        # attributes are df[1:] attributes mapped to set of unique attribute columns
        self.attributes = {}
        self.classifiers = []
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
        # initialize root node with no values
        # calculate split

        main_part = self.train_df

        root = C45_Node(None, None, None, None, False, 0, main_part)

        split_at = best_gain(root, self.attributes)

        # get attribute
        # for each attribute value, append a child node of that value

        d_tree = DecisionTree(root)
        root = d_tree.root

        self.learn_aux(root)

    def learn_aux(self, root):
        # check base cases
        # find attribute split
        # attribute = best_gain(root, self.attributes)

        # root.attribute_name = attribute

        # for value in

        # append new children, recurse on them

        pass

    def entropy(self):

        entropy = 0

        p = 0

        if p != 0:
            entropy += p * math.log(p, 2)

        # from the formula
        entropy = -entropy
        return entropy

    def classify(self):
        pass

    def get_accuracy(self):
        pass


def get_partitions(cand, partition):

    partitions = {}

    for at_val in cand:
        


def best_gain(root, cand_attributes):
    # returns available attribute with best info gain
    entropy = 0
    class_count_dict = root.partition[0].value_counts()
    total_size = class_count_dict[0] + class_count_dict[1]

    candidates = {}
    c_counts = {}

    for C in dict(class_count_dict):
        c_counts[C] = (class_count_dict[C] / total_size)

    for count in c_counts:
        entropy -= c_counts[count] * math.log(c_counts[count], 2)


    for cand in cand_attributes:
        info_gain = 0

        partitions = get_partitions(cand, root.partition)


    # for cand in cand_attributes:
    #         if p != 0:
    #             entropy += p * math.log(p, 2)

    # get count of tuples in D

    # need to errorcheck
    best_at, best_val = max(candidates.items(), key=lambda k: k[1])

    return best_at


def run():
    # get args
    input, support, output = sys.argv[1:]

    print("Complete. Results written to " + "'" + output + "'")


def test(train_set, test_set, out):
    c_45 = C45_Classifier(train_set, test_set, out)

    c_45.preprocess()
    c_45.learn()
    # c_45.entropy()
    # c_45.classify()
    # c_45.get_accuracy()

    print("Complete. Results written to " + "'" + out + "'")


if __name__ == "__main__":
    # check correct length args
    if len(sys.argv) == 1:
        print("No arguments passed, running test mode. Test args: [mushroom.training training.test outputc45.dat]")
        test("mushroom.training", "mushroom.test", "outputc45.dat")
    elif len(sys.argv[1:]) == 3:
        print("Generating results")
        run()
    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")

stop = timeit.default_timer()

print("Results in " + str(stop - start) + " seconds")
