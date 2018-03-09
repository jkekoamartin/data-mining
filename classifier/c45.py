import sys
import numpy
import itertools
from collections import Counter
import pandas as pd
import math

import timeit

start = timeit.default_timer()


# Code written by James Martin

class C45_Node:
    def __init__(self,
                 parent=None,
                 split_attribute=None,
                 classification=None,
                 attribute_value=None,
                 terminal=False,
                 partition=None):
        self.parent = parent
        self.split_attribute = split_attribute
        self.classification = classification
        self.attribute_value = attribute_value
        self.terminal = terminal
        self.partition = partition

        # the list of Nodes with attribute values that correspond to the Nodes attribute name
        self.children = []

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

        # the results of classifting the test set
        self.test_results = []
        self.accuracy = 0.0

    def pre_process(self):
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

    # first call default None
    def learn(self, node=None):

        # if first call, use full dataset
        if node is None:
            node = C45_Node(None, None, None, None, False, self.train_df)
        # new node

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
        node.split_attribute, partitions = best_gain(node, self.attributes)

        self.attributes.pop(node.split_attribute)

        for part in partitions:
            if partitions[part].empty:
                class_counts = dict(node.partition[0].value_counts())
                best_at, best_val = max(class_counts.items(), key=lambda k: k[1])
                leaf = C45_Node(node, None, best_at, part, True, partitions[part])
                node.children.append(leaf)
            else:
                temp_node = C45_Node(node, None, None, part, False, partitions[part])
                new_node = self.learn(temp_node)
                node.children.append(new_node)

        return node

    def classify(self, d_tree):

        test_data = self.test_df

        root = d_tree

        data = []

        nump_arr = test_data.as_matrix()

        for line in nump_arr:
            prediction = classify_tuple(list(line), root)
            line = list(line)
            line.append(prediction)
            data.append(line)

        self.test_results = data

    def get_accuracy(self):

        correct = 0.0
        classified = 0.0

        for line in self.test_results:
            if str(line[0]) is str(line[-1]):
                correct += 1
            else:
                continue
            classified += 1

        accuracy = correct / classified * 100
        self.accuracy = accuracy

    def write(self):

        out = self.output

        out = open(out, 'w')

        output = self.test_results

        for line in output:
            out.write(str(line[1:-1]) + " is classified as: " + str(line[-1]) + ". True class: " + str(line[0]) + "\n")

        out.write("The accuracy is: " + str(self.accuracy) + "%")

        out.close()


def classify_tuple(line, node):
    while node.terminal is False:
        split = node.split_attribute

        for child in node.children:
            if str(line[split]) is str(child.attribute_value):
                node = child
                break
    return node.classification


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

            if dict(partitions[part]):

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
            else:
                part_ratio = 0

            part_entropy += part_ratio * class_entropy

            # get entropy for class labels

        info_gain = (entropy - part_entropy)
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

def run():
    # get args
    training, testing, output = sys.argv[1:]

    c_45 = C45_Classifier(training, testing, output)

    c_45.pre_process()
    root = c_45.learn()
    c_45.classify(root)
    c_45.get_accuracy()
    c_45.write()

    print("Complete. Results written to " + "'" + output + "'")


def test(train_set, test_set, out):
    c_45 = C45_Classifier(train_set, test_set, out)

    c_45.pre_process()
    root = c_45.learn()
    c_45.classify(root)
    c_45.get_accuracy()
    c_45.write()

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
