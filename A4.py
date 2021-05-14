import numpy as np
import pandas as pd
import csv
import itertools

## A.issubset(B)

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
        self.items = set()
        self.importTransactions()
        self.generateUniqueItemSet()

    def run(self):
        """
        Runs the Apriori itemset frequency algorithm and returns a list of sets that pass the minsup, minconf and minlift rules

        """
        k = 1
        # Generate frequent itemsets of length k
        itemsets = self.generateItemSets(self.items, 1)
        frequentSets, infrequentSets = self.eliminateCandidates(itemsets)
        # Repeat until no new frequent itemsets are identified
        while len(frequentSets) != 0:  
            # Generate length (k+1) candidate itemsets from length k frequent itemsets
            k += 1
            itemsets = self.generateItemSets(self.items, k)
            # Prune candidate itemsets containing subsets of length k that are infrequent
            itemsets = self.prune(itemsets, infrequentSets)
            # Count the support of each candidate by scanning the DB
            # Eliminate candidates that are infrequent, leaving only those that are frequent
            backup = frequentSets.copy()
            frequentSets, infrequentSets = self.eliminateCandidates(itemsets)
        frequentSets = backup

        # From frequent sets find rules that meet the confidence threshold
        rules = []
        for rule in frequentSets:
            k = len(rule)
            # generate subsets from 1 to k-1 in length
            for num in range(1, k):
                itemsets = self.generateItemSets(rule, num)
                for itemset in itemsets:
                    if self.calculateConfidence(rule, itemset) > self.minconf: # TODO: Implement LIFT check
                        rules.append(itemset)

        return rules
    
    def eliminateCandidates(self, itemsets:list)->tuple:
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
        Generates combinations of items of a certain lengths and returns a list of sets

        Parameters
        items (set): The set of items to generate combinations from
        k (int): The length of the generated sets

        Returns:
        list: List of the generated combinations of length k
        """
        itemsets = list(itertools.combinations(items, k))
        for i in range(len(itemsets)):
            itemsets[i] = set(itemsets[i])
        return itemsets

    def generateUniqueItemSet(self)->set:
        """
        Generates a set of items that appear across all transactions
        """
        for transaction in self.transactions:
            for item in transaction:
                if item not in self.items:
                    self.items.add(item)
        return self.items

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

    def calculateSupportValues(self, itemsets:list)->list:
        """
        Gets a list for the support values for each itemset in a list of itemsets

        Parameters:
        itemsets (list): List of itemsets

        Returns
        list: A list containing float values that correspond to the sets in itemsets
        """
        supportValues = []
        for itemset in itemsets:
            support = self.calculateSupport(itemset)
            supportValues.append(support)
        return supportValues

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
        list: A pruned list of sets that do not contain of the infrequentSets as subsets
        """
        prunedSets = []
        if len(infrequentSets) == 0:
            return itemsets
        else:
            for itemset in itemsets:
                for infrequentSet in infrequentSets:
                    if (not infrequentSet.issubset(itemset)) and (itemset not in prunedSets):
                        prunedSets.append(itemset)
            return prunedSets


def main():
    path = 'transactions.csv'
    apriori = Apriori(0.15, 0.8, 0.1, path)
    frequentSets = apriori.run()

    print(frequentSets)

if __name__ == "__main__":
    main()