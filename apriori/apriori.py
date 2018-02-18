import sys
import itertools
from collections import Counter
import timeit

start = timeit.default_timer()


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
                if isinstance(key, tuple):
                    print(" ".join(map(str, list(key))) + " (" + str(dict_set[key]) + ")")
                else:
                    print(str(key) + " (" + str(dict_set[key]) + ")")


class Apriori:

    def __init__(self, read, write, min_sup):

        self.read = read
        self.write = write
        self.min_sup = int(min_sup)

        data = open(read)
        unpruned_list = []

        for line in data:
            unpruned_list += [line.split()]

        # Object to store len-k frequent sets.
        self.frequent_sets = FrequentSets(unpruned_list)

    def gen_k_sets(self):

        # generate each k itemset
        self.gen_k_1()
        self.gen_k_2()
        self.gen_k_nth()

    def gen_k_1(self):

        frequent_sets = self.frequent_sets

        # raw list from frequent sets object
        unpruned_list = frequent_sets.raw_list

        unpruned_dict = {}

        for row in unpruned_list:
            for column in row:
                if column in unpruned_dict:
                    unpruned_dict[column] += 1
                else:
                    unpruned_dict[column] = 1

        # prune is set of keys to be kept in raw list
        prune_set = self.prune(unpruned_dict)

        # able to use set operations without duplicate removal thanks to Counter()!

        self.prune_list(prune_set, unpruned_list)

    def gen_k_2(self):

        frequent_sets = self.frequent_sets

        perm_dict = {}
        raw_list = []

        unpruned_list = frequent_sets.raw_list[:]
        for row in unpruned_list:
            perms = list(itertools.combinations(row, 2))
            raw_list.append(perms)

            for item in perms:
                if item in perm_dict:
                    perm_dict[item] += 1
                else:
                    perm_dict[item] = 1

        prune_set = self.prune(perm_dict)

        self.prune_list(prune_set, raw_list)

    def gen_k_nth(self):

        frequent_sets = self.frequent_sets

        empty = False
        k = 3

        while not empty:
            new_raw = []
            perm_dict = {}
            raw_list = frequent_sets.raw_list

            for row in raw_list:
                # chains tuples together, like self-joining sets
                row = Counter(itertools.chain(*row))
                # sort, or a few tuples will get confused by the counter
                row = sorted(list(row))
                perms = list(itertools.combinations(row, k))
                new_raw.append(perms)

                for item in perms:
                    if item in perm_dict:
                        perm_dict[item] += 1
                    else:
                        perm_dict[item] = 1

            prune_set = self.prune(perm_dict)

            self.prune_list(prune_set, new_raw)

            if not prune_set:
                empty = True
            # update loop vars
            k += 1

        print("Formatting Output")

    def prune(self, unpruned_dict):

        prune = []
        pruned_dict = unpruned_dict.copy()

        for key in unpruned_dict:
            if unpruned_dict[key] >= self.min_sup:
                prune.append(key)
            else:
                pruned_dict.pop(key)

        # add dictionary to frequent_sets object
        self.frequent_sets.add_dict(pruned_dict)

        # returning prune set. prune set to clean raw list
        return set(prune)

    def prune_list(self, prune_set, raw_list):

        new_raw_list = []

        for line in raw_list:
            new_line = [x for x in line if x in prune_set]
            if new_line:
                new_raw_list.append(new_line)

        self.frequent_sets.raw_list = new_raw_list

    def write_output(self):

        out = self.write

        out = open(out, 'w')

        output = self.frequent_sets.k_sets_list

        for dict_set in output:
            for key in dict_set:
                if isinstance(key, tuple):
                    out.write(" ".join(map(str, list(key))) + " (" + str(dict_set[key]) + ")\n")
                else:
                    out.write(str(key) + " (" + str(dict_set[key]) + ")\n")

        out.close()


def run():
    # get args
    input, support, output = sys.argv[1:]
    # init Apriori
    apriori = Apriori(input, output, support)
    # gen results
    apriori.gen_k_sets()
    # write results to file
    apriori.write_output()

    print("Complete. Results written to " + "'" + output + "'")


def test(inp, sup, out):
    input = inp
    support = sup
    output = out

    # init Apriori
    apriori = Apriori(input, output, support)
    # gen results
    apriori.gen_k_sets()
    # print results
    apriori.frequent_sets.to_string()
    # write results
    apriori.write_output()
    print("Complete. Results written to " + "'" + output + "'")


if __name__ == "__main__":
    # check correct length args
    if len(sys.argv) == 1:
        print("No arguments passed, running test mode. Test args: [T10I4D100K.dat 500 output500.dat]")
        test("T10I4D100K.dat", 500, "output500.dat")
    elif len(sys.argv[1:]) == 3:
        print("Generating results")
        run()

    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")

stop = timeit.default_timer()

print("Results in " + str(stop - start) + " seconds")
