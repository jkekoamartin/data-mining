import sys

import itertools


class Apriori:

    def __init__(self, read, write, min_sup, cand_dict={}):
        self.cand_dict = cand_dict
        self.read = read
        self.write = write
        self.min_sup = min_sup

    def gen_cand(self, read):

        # brute force

        candidates = []
        data = open(read)
        lines = []

        for line in data:
            lines += [line.split()]

        # gen cands two levels, filter down using apriori property
        # 'T10I4D100K.dat' is stored in list of lists, where each index in 'ln' is line of 'T10I4D100K.dat'

        fpm = {}

        for row in lines:
            for column in row:
                if column not in fpm.keys():
                    fpm[column] = 1
                else:
                    fpm[column] += 1

        # need to filter cand_set by prune set of keys, then count again
        fpm, prune = self.prune(fpm)

        c = 0
        # for x in fpm.keys():
        #     c += 1
        #     print(str(x) + " (" + str(fpm[x]) + ")")

        temp = lines[:]

        nelines =[]

        for line in lines:
            line = [x for x in line if x not in prune]
            nelines.append(line)

        for x, y in zip(nelines, temp):
                print(str(len(x)) + " " + (str(len(y))))

        # count = c
        # while count != 0:
        #
        #     lines = self.clean(lines, prune)
        #
        #     for x in fpm.keys():
        #         c += 1
        #         print(str(x) + " (" + str(fpm[x]) + ")")

        # need to generate candidates
        # count = {}
        # for line in lines[:11]:
        #     # print(key)
        #     temp = count[:]
        #     for item in line:
        #         if item not in list(temp.keys()):
        #             temp[item] = 1
        #         else:=
        #             new_val = temp[item]
        #             new_val += 1
        #             temp[item] = new_val
        #     count = temp
        # for x in count.keys():
        #     if count[x] < 3:
        #         count.pop(x)
        #         print("removed: " + str(x))

        # useful!
        print(list(itertools.permutations(lines[0], 2)))

        return candidates

    def prune(self, cands):
        prune = []
        for x in cands.keys():
            if cands[x] < self.min_sup:
                prune.append(x)

        for p in prune:
            cands.pop(p)

        return cands, prune

    def count_fp(self, read, candidates):

        # count candidates, add to dict if cand is in dict,
        results = []
        fpm = {}
        unique = 0

        ln = []
        data = open(read)

        for line in data:
            ln += [line.split()]

        for row in ln:
            for column in row:
                if column not in fpm.keys():
                    fpm[column] = 1
                    unique += 1
                else:
                    fpm[column] += 1

        # for x in fpm.keys():
        #     print(str(x) + " (" + str(fpm[x]) + ")")

        return results

    def write_output(self, result, write):
        print("Need to implement write to file")


def run():
    input, support, output = sys.argv[1:]

    apriori = Apriori(input, output, support)
    cands = apriori.gen_cand(apriori.read)
    result = apriori.count_fp(apriori.read, cands)

    apriori.write_output(result, apriori.write)


def test(inp, sup, out):
    input = inp
    support = sup
    output = out

    apriori = Apriori(input, output, support)
    cands = apriori.gen_cand(apriori.read)
    result = apriori.count_fp(apriori.read, cands)

    apriori.write_output(result, apriori.write)


if __name__ == "__main__":

    # check correct length args
    if len(sys.argv) == 1:
        print("test output")
        test("T10I4D100K.dat", 500, "output.dat")
    elif len(sys.argv[1:]) == 3:
        run()
    else:
        print("Invalid number of arguments passed. Please input: [Readfile MinimumSupport OutputFile]")
