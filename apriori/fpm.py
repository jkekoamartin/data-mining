# The program should be executable with 3 parameters: the name of the input dataset file, the threshold of minimum
# support count, and the name of the output file (in that order).  The minimum support count should be an integer.
# An itemset is frequent if its support count is larger or equal to this threshold.
#
# The program should output a file that contains all the frequent itemsets together with their support.  The output
# file (sample output) should have the following format: each line contains a single frequent itemset as a list of
# items separated by whitespace.  At the end of the line, its support is printed between a pair of parenthesis.  For
# example: 1 2 3 (5) represents an itemset containing items 1, 2 and 3 with a support count of 5.

# read data


#
# # 'T10I4D100K.dat' is stored in list of lists, where each index in 'ln' is line of 'T10I4D100K.dat'
# #
# # run alg
# fpm = {}
# unique = 0
# for row in ln:
#     for column in row:
#         if column not in fpm.keys():
#             fpm[column] = 1
#             unique += 1
#         else:
#             fpm[column] += 1
#
# # save output to file
# print(str(unique) + " unique values")
#
# for x in fpm.keys():
#     print(str(x) + " (" + str(fpm[x]) + ")")

# *************Apriori Algorithm*************
# Zealseeker | www.zealseeker.com
# *******************************************
# Input:
#   dataDic, database of events as a dict
#       -----------------
#       |TID    ItemID  |
#       |T100   I1,I2,I5|
#       |...    ...     |
#       -----------------
#   min_sup, minimize support threshold
# Output:
#   UkLk, set of frequent(large) itemsets,
# *******************************************
import itertools


class Apriori:
    def __init__(self, min_sup=0.2, dataDic=None):
        if dataDic is None:
            dataDic = {}
        self.data = dataDic
        self.size = len(dataDic)  # Get the number of events
        self.min_sup = min_sup
        self.min_sup_val = 500

    def find_frequent_1_itemsets(self):
        FreqDic = {}  # {itemset1:freq1,itemsets2:freq2}
        for event in self.data:
            for item in self.data[event]:
                if item in FreqDic:
                    FreqDic[item] += 1
                else:
                    FreqDic[item] = 1
        L1 = []

        for itemset in FreqDic:
            if FreqDic[itemset] >= self.min_sup_val:
                L1.append([itemset])
        return L1

    def has_infrequent_subset(self, c, L_last, k):
        subsets = list(itertools.combinations(c, k - 1))  # return list of tuples of items
        for each in subsets:
            each = list(each)  # change tuple into list
            if each not in L_last:
                return True
        return False

    def apriori_gen(self, L_last):  # L_last means frequent(k-1) itemsets
        k = len(L_last[0]) + 1
        Ck = []
        for itemset1 in L_last:
            for itemset2 in L_last:
                # join step
                flag = 0
                for i in range(k - 2):
                    if itemset1[i] != itemset2[i]:
                        flag = 1  # the two itemset can't join
                        break
                if flag == 1: continue
                if itemset1[k - 2] < itemset2[k - 2]:
                    c = itemset1 + [itemset2[k - 2]]
                else:
                    continue

                # pruning setp
                if self.has_infrequent_subset(c, L_last, k):
                    continue
                else:
                    Ck.append(c)
        return Ck

    def do(self):
        L_last = self.find_frequent_1_itemsets()
        L = L_last
        i = 0
        while L_last != []:
            Ck = self.apriori_gen(L_last)
            FreqDic = {}
            for event in self.data:
                # get all suported subsets
                for c in Ck:
                    if set(c) <= set(self.data[event]):  # is subset
                        if tuple(c) in FreqDic:
                            FreqDic[tuple(c)] += 1
                        else:
                            FreqDic[tuple(c)] = 1
            Lk = []
            for c in FreqDic:
                if FreqDic[c] > self.min_sup_val:
                    Lk.append(list(c))
            L_last = Lk
            L += Lk
            print("hello")
            print(L)
        return L


# ******Test******
# data = open('T10I4D100K.dat')
# ln = []
#
# for line in data:
#     ln += [line.split()]
#
# Data = {}
# key = 100
# for line in ln:
#     Data['T'+str(key)] = line
#     key += 100

Data = {'T100': ['I1', 'I2', 'I5'],
        'T200': ['I2', 'I4'],
        'T300': ['I2', 'I3'],
        'T400': ['I1', 'I2', 'I4'],
        'T500': ['I1', 'I3'],
        'T600': ['I2', 'I3'],
        'T700': ['I1', 'I3'],
        'T800': ['I1', 'I2', 'I3', 'I5'],
        'T900': ['I1', 'I2', 'I3']}

a = Apriori(dataDic=Data)
a.do()
