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

        frequent_sets = self.frequent_sets

        data = open(read)
        lines = []

        for line in data:
            lines += [line.split()]

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

        new_raw_list = []

        prune = self.prune(unpruned_dict)

        # able to use set operations without duplicate removal thanks to Counter()!
        for line in self.frequent_sets.raw_list:
            new_raw_list.append(Counter(line) & Counter(prune))

        #   update object
        self.frequent_sets.raw_list = new_raw_list[:]

        # k =2

        perm_dict = {}

        new_raw_list = []

        for row in self.frequent_sets.raw_list:
            perms = list(itertools.combinations(row, 2))
            new_raw_list.append(perms)

            for item in perms:
                if item in perm_dict.keys():
                    perm_dict[item] += 1
                else:
                    perm_dict[item] = 1

        prune = self.prune(perm_dict)

        # able to use set operations without duplicate removal thanks to Counter()!
        for line in self.frequent_sets.raw_list:
            new_raw_list.append(Counter(line) & Counter(prune))

        #   update object
        self.frequent_sets.raw_list = new_raw_list[:]

        self.apriori()

        # return object
        return frequent_sets

    def prune(self, cands):

        prune = []
        temp = cands.copy()

        count = 0
        for key in cands:
            if cands[key] < self.min_sup:
                temp.pop(key)
            else:
                prune.append(key)
            count += 1

        self.frequent_sets.add_dict(temp)
        return prune
        # returning frequent sets dictionary and prune set. prune set to clean raw list

    def apriori(self):

        k = 3

        while k < 6:
            raw = self.frequent_sets.raw_list

            self.permute(raw, k)

            k += 1

        print("Formatting Output")

    def permute(self, raw, k):
        perm_dict = {}

        new_raw_list = []

        for row in raw:
            row = Counter(itertools.chain(*row))
            perms = list(itertools.combinations(row, k))
            # instead, merge existing tuples, via k
            new_raw_list.append(perms)

            # maybe optimize this? like after 1000 sweeps, stop checking
            for item in perms:
                if item in perm_dict.keys():
                    perm_dict[item] += 1
                else:
                    perm_dict[item] = 1

        new_raw_list = []

        prune = self.prune(perm_dict)
        for line in self.frequent_sets.raw_list:
            new_raw_list.append(Counter(line) & Counter(prune))

        #   update object
        self.frequent_sets.raw_list = new_raw_list[:]



def test(inp, sup, out):
    input = inp
    support = sup
    output = out

    apriori = Apriori(input, output, support)
    result = apriori.gen_cand(apriori.read)
    result.to_string()


def run():
    input, support, output = sys.argv[1:]

    apriori = Apriori(input, output, support)
    cands = apriori.gen_cand(apriori.read)


if __name__ == "__main__":
    # check correct length args
    if len(sys.argv) == 1:
        print("Processing Input: 12 Minutes for 10000 line dataset")
        test("T10I4D100K.dat", 3, "output.dat")
    elif len(sys.argv[1:]) == 3:
        run()
    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")
