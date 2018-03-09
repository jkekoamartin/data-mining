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
                 split_value=None,
                 partition=None):
        # self.attribute_split_index = attribute_split_index
        self.split_value = split_value
        self.partition = partition
        self.attribute_name = attribute_name

        self.parent = parent

        # the list of Nodes with attribute values that correspond to the Nodes attribute name
        self.children = []

        self.classification = classification

        # this is the attribute value that link the node to the parent
        self.attribute_value = attribute_value

        # flag for if terminal, should have a classification if terminal
        self.terminal = terminal

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class C45_Classifier:

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
        root = self.learn_aux()

        print(root)

    # first call default None
    def learn_aux(self, root=None):

        # if first call, use full dataset
        if root is None:
            node = C45_Node(None, None, None, None, False, None, self.train_df)
        # new node
        else:
            node = C45_Node(root, None, None, None, False, None, self.train_df)
        # check base cases
        class_counts = dict(node.partition[0].value_counts())

        # is this node partition fully classified?
        # assign leaf
        if len(class_counts) is 1:
            node.classification = list(class_counts.keys()).pop()
            node.terminal = True
            return node
        # if attributes list is empty, cant split, so classify the majority

        # find attribute split, assign. get partitions, this also remove attribute from list
        node.split_value, partitions = best_gain(node, self.attributes)


        # remove from list
        print(node.split_value)

        if node.split_value == 0:
            best_at, best_val = max(class_counts.items(), key=lambda k: k[1])
            node.classification = best_at
            node.terminal = True
            return node

        self.attributes.pop(node.split_value)

        for part in partitions:
            if partitions[part].empty:
                class_counts = dict(root.partition[0].value_counts())
                best_at, best_val = max(class_counts.items(), key=lambda k: k[1])
                leaf = C45_Node(node, node.split_value, best_at, part, True, None, partitions[part])
                node.children.append(leaf)
            else:
                temp_node = C45_Node(node, node.split_value, None, part, False, None, partitions[part])
                new_node = self.learn_aux(temp_node)
                node.children.append(new_node)

        return node

    def classify(self):
        pass

    def get_accuracy(self):
        pass


def best_gain(root, cand_attributes):
    # returns available attribute with best info gain
    entropy = 0

    class_count_dict = dict(root.partition[0].value_counts())
    total_size = sum(class_count_dict.values())
    # candidates = {}
    class_size = {}

    # cast to dict, to wrangle with pandas dataframe

    for c_label in class_count_dict:
        class_size[c_label] = (class_count_dict[c_label] / total_size)

    # get entropy for class labels
    for count in class_size:
        entropy -= class_size[count] * math.log(class_size[count], 2)

    # cand_att is a dict, for each attribute, calculate info gain on partition, return best one

    # [attribute, dictionary of it's resulting partition, it's info gain]
    candidate_container = [0, dict(), 0]

    for attribute in cand_attributes:
        # this has the x,y,z partition index by x,y,z
        partitions = get_partitions(attribute, cand_attributes[attribute], root.partition)

        # 0 mean that partitions perfectly classify
        part_entropy = 0
        # for each partition calculate entropy then normalize, add to, remember, this traverses a dict
        for part in partitions:

            class_entropy = 0

            att_count_dict = dict(partitions[part][attribute].value_counts())
            part_total_size = sum(att_count_dict.values())
            part_class_dict = dict(partitions[part][0].value_counts())
            part_class_size = {}

            # cast to dict, to wrangle with pandas dataframe

            # store normalized probability
            part_ratio = (part_total_size / total_size)

            for c_label in part_class_dict:
                part_class_size[c_label] = (part_class_dict[c_label] / part_total_size)

            for count in part_class_size:
                class_entropy -= part_class_size[count] * math.log(part_class_size[count], 2)

            part_entropy += part_ratio * class_entropy
            # get entropy for class labels

        info_gain = entropy - part_entropy
        if info_gain > candidate_container[2]:
            candidate_container = [attribute, partitions, info_gain]

    # trim info gain off, was only neccesary for getting max
    return candidate_container[:2]


def get_partitions(cand, cand_values, partition):
    # this return the partitions from splitting on atrribute values, used to calc info gain
    partitions = {}

    temp = []
    nump_array = partition.as_matrix()

    for value in cand_values:
        for row in nump_array:
            if row[cand] is value:
                temp.append(row)

        partitions[value] = pd.DataFrame(temp)
        temp.clear()

    # return dictionary of attribute values mapped to partition that value split
    return partitions


# todo: fix
# def print_tree(node):
#     if node.is_leaf_node:
#         classification = node.classification
#         print('Found a leaf! This leaf is predicted to be: ' + classification)
#         return
#     elif node.parent is None:
#         print('This is the root node. splitting on ' + str(node.attribute_split_column))
#         print("\n")
#         index = 0
#         for character in node.attribute_split_character_list:
#             print('Traversing to ' + character)
#             print_tree(node.children[index])
#             index += 1
#     elif node.parent is not None:
#         print('This node in the tree will split on ' + str(node.attribute_split_column))
#         index = 0
#         for character in node.attribute_split_character_list:
#             print('Traversing to ' + character)
#             print_tree(node.children[index])
#             index += 1

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
