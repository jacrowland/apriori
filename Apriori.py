import numpy as np
import pandas as pd
import csv
import itertools
import time

class Apriori():
    """
    This object implements the Apriori unsupervised machine learning algorithm 
    for frequent item set mining and association rule learning
    """
    def __init__(self, minsup:float, minconf:float, minlift:float, path:str):
        """Initalise the Apiori class

        Parameters:
        minsup (float): Minimuim support
        minconf (float): Minimum confidence
        minlift (float): Minimum lift
        path (str): Transaction CSV path - each line is transaction of items seperated by comma
        items (set): The set of unique items found across all transactions - initalised by generateUniqueItemSet()
        transactions (list): A list of sets for each transaction - where each is a set of items

        Returns:
        None:Returning value
        """
        self.minsup = minsup
        self.minconf = minconf
        self.minlift = minlift
        self.path = path
        self.transactions = []
        self.items = []
        self.importTransactions()
        self.generateUniqueItemSet()

    def run(self):
        """
        Runs the Apriori itemset frequency algorithm and returns a list of sets that pass the minsup, minconf and minlift rules
        """
        print("Finding associations rules from {} transactions and {} unique items...".format(len(self.transactions), len(self.items)))

        frequentSets = self.generateFrequentSets()
        associationRules = self.generateAssociationRules(frequentSets)
        #associationRules = self.sortAssociationRules(associationRules) # sorts rules by length, lift, confidence and support

        #frequentSets = [{'BREAD AND CAKE', 'BISCUITS', 'FRUIT', 'BAKING NEEDS', 'FROZEN FOODS', 'VEGETABLES'}]

        return associationRules

    def generateAssociationRules(self, frequentSets:list)->list:
        """
        Generates a list of association rules that meat the confidence and lift thresholds

        Parameters:
        frequentSets (list): A list of frequent itemsets

        Returns:
        list: A list of containing the association rules that pass the confidence and lift thresholds
        """
        # From frequent sets find rules that meet the confidence threshold
        associationRules = []
        for rule in frequentSets:
            itemsets = []
            for item in rule:
                s = set()
                s.add(item)
                itemsets.append(s)
            # generate subsets from 1 to k-1 in length
            for num in range(2, len(itemsets)):
                itemsets = self.generateItemSets(itemsets, num)
                for itemset in itemsets:
                    if self.calculateConfidence(rule, itemset) > self.minconf: # TODO: Implement LIFT check
                        associationRules.append(itemset)
        return associationRules

    def generateFrequentSets(self)->list:
        """
        Implements the Apriori frequent itemset generation algorithm.
        Where itemsets is c_k and frequentSets = l_k 

        Returns:
        list: A list of frequent itemsets of size k-1
        """
        # Generate frequent itemsets of length k
        frequentSets, infrequentSets = self.eliminateCandidates(self.items) # c_1
        # Repeat until no new frequent itemsets are identified
        k = 1
        while True:  
            k += 1
            print("k: " + str(k) + "    " , end="\n")
            # Generate length (k+1) candidate itemsets from length k frequent itemsets
            itemsets = self.generateItemSets(frequentSets, k)
            # Prune candidate itemsets containing subsets of length k that are infrequent
            itemsets = self.prune(itemsets, infrequentSets)
            # Count the support of each candidate by scanning the DB
            # Eliminate candidates that are infrequent, leaving only those that are frequent
            prevFrequentSets = frequentSets.copy() # holds a copy of frequent sets for k-1
            frequentSets, infrequentSets = self.eliminateCandidates(itemsets)
            # Runs until no frequent itemsets are identified
            if len(frequentSets) == 0:
                frequentSets = prevFrequentSets
                break
        return frequentSets

    def sortAssociationRules(self, associationRules:list)->list: # TODO: sorted output list
        pass
    
    def eliminateCandidates(self, itemsets:list)->tuple:
        """
        Sorts candiadate itemsets by calculating the support value and comparing to the mininimum support value

        Parameters:
        itemsets (list): A list of sets to sort

        Returns:
        tuple: Returns a tuple of two sets. Containing the eliminated rules and one containing rules that pass
        """
        frequentSets = []
        infrequentSets = []
        for i in range(len(itemsets)):
            if not self.calculateSupport(itemsets[i]) < self.minsup:
                frequentSets.append(itemsets[i])
            else:
                infrequentSets.append(itemsets[i])
        return frequentSets, infrequentSets

    def generateItemSets(self, items:set, k:int)->list:
        """
        Generates combinations of itemsets and returns a list of sets

        Parameters
        items (set): The set of items to generate combinations from
        k (int): The length of the generated sets

        Returns:
        list: List of the generated combinations of length k
        """
        pairs = list(itertools.combinations(items, 2))

        itemsets = []
        for pair in pairs:
            set1 = pair[0]
            set2 = pair[1]
            set3 = set1.union(set2)
            if set3 not in itemsets and len(set3) == k:
                itemsets.append(set3)
        return itemsets

    def generateUniqueItemSet(self):
        """
        Generates a set of items that appear across all transactions. Updates the class attribute self.transactions
        """
        items = set()
        for transaction in self.transactions:
            for item in transaction:
                if item not in self.items:
                    items.add(item)
        for item in items:
            s = set()
            s.add(item)
            self.items.append(s)

    def importTransactions(self):
        """
        Reads in the csv file of transactions. The file is assumed to have no header. Each row is a set of items contained within a transaction
        """
        with open(self.path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                cleanedRow = []
                for item in row:
                    cleanedRow.append(item.strip().upper())
                self.transactions.append(set(cleanedRow))

    def calculateSupport(self, itemset:set)->float:
        """
        Calculates the support value of a set by dividing the number of transactions a set occurs in with the total number of transactions

        Parameters
        itemset (set): A set of items

        Returns:
        float: The support value for that set (rule)

        """
        return self.count(itemset) / len(self.transactions)

    def calculateConfidence(self, s:set, i:set)->float:
        """
        Calculates how often items in Y appear in transactions containing X

        Parameters
        set (set): A set of items

        Returns
        float: The confidence value for that set (rule)

        """
        return self.calculateSupport(s) / self.calculateSupport(i)

    def calculateLift(self, body:list, head:list)->float:
        """
        Calculates the importance of a rule

        """
        pass

    def count(self, s:set)->int:
        """
        Parameters
        set (set): A set of items

        Returns:
        int: Frequency count for how many times s is a subset of a transaction t
        """
        count = 0
        for transaction in self.transactions:
            if set(s).issubset(transaction):
                count += 1
        return count

    def prune(self, itemsets:list, infrequentSets:list)->list:
        """
        Prunes itemsets from the list of itemsets if any of the sets in infrequentSets are subsets of the itemset

        Parameter
        itemsets (list): A list of itemsets to prune
        infrequentSets (list): A list of subsets to check

        Returns
        list: A pruned list of itemsets
        """
        if len(infrequentSets) == 0:
            return itemsets
        else:
            prunedSets = []

            for i in range(len(itemsets)):
                print(str(round(i / len(itemsets) * 100, 2)) + "%", end="")
                print("\r", end="")
                for j in range(len(infrequentSets)):
                    if (not infrequentSets[j].issubset(itemsets[i])) and (itemsets[i] not in prunedSets):
                        prunedSets.append(itemsets[i])
            return prunedSets


def main():
    startTime = time.time()

    path = 'supermarket.csv'
    apriori = Apriori(minsup=0.15, minconf=0.8, minlift=1, path=path)
    frequentSets = apriori.run()
    print(frequentSets)
    print()
    print(round(time.time() - startTime), end=" seconds")

if __name__ == "__main__":
    main()