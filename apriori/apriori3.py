import sys
import itertools
from collections import Counter


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
                if not isinstance(key, tuple):
                    print(str(key) + " (" + str(dict_set[key]) + ")")
                else:
                    print(str(key) + " (" + str(dict_set[key]) + ")")

                    # print(" ".join(key) + " (" + str(dict_set[key]) + ")")


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
                if column in unpruned_dict.keys():
                    unpruned_dict[column] += 1
                else:
                    unpruned_dict[column] = 1

        # need to filter cand_set by prune set of keys, then count again
        # prune is set of keys to be removed from raw list

        pruned_dict, prune_set = self.prune(unpruned_dict)

        new_raw_list = []

        # able to use set operations without duplicate removal thanks to Counter()!
        for line in lines:
            new_raw_list.append(Counter(line) & Counter(prune_set))
        # update object
        frequent_sets.add_dict(pruned_dict)
        frequent_sets.raw_list = new_raw_list[:]


        # k = 2
        perm_dict = {}
        raw_list = []

        raw = frequent_sets.raw_list
        for row in raw:
            perms = list(itertools.combinations(row, 2))
            raw_list.append(perms)

            # maybe optimize this? like after 1000 sweeps, stop checking
            for item in perms:
                if item in perm_dict.keys():
                    perm_dict[item] += 1
                else:
                    perm_dict[item] = 1

        pruned_dicts, prune_set = self.prune(perm_dict)

        new_raw_list = []
        for line in raw_list:
            new_raw_list.append(Counter(line) & Counter(prune_set))
        # update object
        frequent_sets.add_dict(pruned_dicts)
        frequent_sets.raw_list = new_raw_list[:]

        self.apriori(frequent_sets)

        # return object
        return frequent_sets

    def apriori(self, frequent_sets):



        k = 3

        while k < 6:
            new_raw = []
            perm_dict = {}
            raw = frequent_sets.raw_list
            for row in raw:
                row = Counter(itertools.chain(*row))
                perms = list(itertools.combinations(row, k))
                new_raw.append(perms)
                # maybe optimize this? like after 1000 sweeps, stop checking
                for item in perms:
                    if item in perm_dict.keys():
                        perm_dict[item] += 1
                    else:
                        perm_dict[item] = 1

            pruned_dict, prune_set = self.prune(perm_dict)

            new_raw_list = []
            for line in raw:
                new_raw_list.append(Counter(line) & Counter(prune_set))

            # update object

            frequent_sets.add_dict(pruned_dict)
            frequent_sets.raw_list = new_raw
            k += 1

        print("Formatting Output")
        # for line in curr_list:

    def prune(self, cands):

        prune = []
        temp = cands.copy()

        for key in cands:
            if cands[key] >= self.min_sup:
                prune.append(key)
            else:
                temp.pop(key)


        # returning frequent sets dictionary and prune set. prune set to clean raw list
        return temp, prune

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
        print("Processing Input: 12 Minutes for 10000 line dataset")
        test("T10I4D100K.dat", 500, "output.dat")
    elif len(sys.argv[1:]) == 3:
        run()
    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")
