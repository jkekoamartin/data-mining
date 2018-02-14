import sys

import itertools


class FrequentSets:

    def __init__(self, raw):

        # raw list in process
        self.raw_list = raw

        # results, stored as list of dictionaries of k-len sets
        self.k_sets_list = []

    def add_dict(self, frequent_set):
        self.k_sets_list.append(frequent_set)

    def to_string(self):
        out = self.k_sets_list

        for dict_set in out:
            for key in dict_set:
                print(str(key) + " (" + str(dict_set[key]) + ")")


class Apriori:

    def __init__(self, read, write, min_sup):

        self.read = read
        self.write = write
        self.min_sup = min_sup

        # Object to store len-k frequent sets.

        lines = [
            [1, 2, 3, 4, 5, 6],
            [7, 2, 3, 4, 5, 6],
            [1, 8, 4, 5],
            [1, 9, 0, 4, 6],
            [0, 2, 2, 4, 5],
        ]

        self.frequent_sets = FrequentSets(lines)

    def gen_cand(self, read):

        # dynamic code, uncomments after test

        data = open(read)
        lines = []

        for line in data:
            lines += [line.split()]

        # gen cands two levels, filter down using apriori property
        # 'T10I4D100K.dat' is stored in list of lists, where each index in 'ln' is line of 'T10I4D100K.dat'

        frequent_sets = self.frequent_sets

        # raw list from frequent sets object
        # lines = frequent_sets.raw_list

        unpruned_dict = {}

        # count k = 1
        for row in lines:
            for column in row:
                if column not in unpruned_dict.keys():
                    unpruned_dict[column] = 1
                else:
                    unpruned_dict[column] += 1

        # need to filter cand_set by prune set of keys, then count again
        # prune is set of keys to be removed from raw list

        pruned_dict, prune_set = self.prune(unpruned_dict)

        new_raw_list = []

        # this uses set for sake of speed. This remove duplicates unfortunately, which may not work with other data sets
        # todo: look for solution other than set subtraction
        count = 0

        for line in lines:
            # print("ocrapnoodles" + str(count) + " " + str(len(line)))
            new_raw_list.append(set(line)-set(prune_set))
            count += 1
            # line = [x for x in line if x not in prune_set]
            # new_raw_list.append(line)
        # update object

        frequent_sets.add_dict(pruned_dict)
        frequent_sets.raw_list = new_raw_list

        self.apriori(frequent_sets)

        # return object
        return frequent_sets

    def apriori(self, frequent_sets):

        raw = frequent_sets.raw_list

        for x in raw:
            print(x)

        k = 2

        while k < 3:
            perm_dict = {}

            for row in raw:
                perms = list(itertools.combinations(row, k))
                for item in perms:
                    if item not in perm_dict.keys():
                        perm_dict[item] = 1
                    else:
                        perm_dict[item] += 1

            pruned_dict, prune_set = self.prune(perm_dict)

            new_raw_list = []

            count = 0
            for line in raw:
                # print("ocrapnoodles" + str(count))
                new_raw_list.append(set(line) - set(prune_set))
                count += 1

            # update object
            frequent_sets.add_dict(pruned_dict)
            frequent_sets.raw_list = new_raw_list
            k += 1

        # for line in curr_list:

    def prune(self, cands):

        prune = []

        count = 0
        print("hey!"+str(len(cands)))
        for key in cands:
            if cands[key] < self.min_sup:
                prune.append(key)
                count += 1
        for key in prune:
            cands.pop(key)

        # returning frequent sets dictionary and prune set. prune set to clean raw list
        return cands, prune

    def write_output(self, result, write):
        print("Need to implement write to file")


def run():
    input, support, output = sys.argv[1:]

    apriori = Apriori(input, output, support)
    cands = apriori.gen_cand(apriori.read)
    # result = apriori.count_fp(apriori.read, cands)
    #
    # apriori.write_output(result, apriori.write)


def test(inp, sup, out):
    input = inp
    support = sup
    output = out

    apriori = Apriori(input, output, support)
    result = apriori.gen_cand(apriori.read)
    result.to_string()
    # result = apriori.count_fp(cands)
    #
    # apriori.write_output(result, apriori.write)


if __name__ == "__main__":
    # check correct length args
    if len(sys.argv) == 1:
        print("test output")
        test("T10I4D100K.dat", 500, "output.dat")
    elif len(sys.argv[1:]) == 3:
        run()
    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")
