import csv
import itertools
import time
import pandas as pd
from AssociationRule import AssociationRule
class Apriori():
    """
    An implementation of the Apriori unsupervised machine learning algorithm for frequent item set mining and association rule learning
    
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
        Runs the Apriori itemset frequency algorithm and returns a sorted list of association rules that pass the minsup, minconf and minlift rules
        """
        print("Running Apriori...")
        print(" a. Finding association rules from {} transactions and {} unique items...".format(len(self.transactions), len(self.items)))
        frequentSets = self.generateFrequentSets() # find all frequent itemsets of length 1 to k-1
        print(" b. Generating association rules from frequent itemsets...")
        associationRules = self.generateAssociationRules(frequentSets) # generate rules that pass the minsup, minlift and minsup thresholds
        print(" c. Sorting association rules...")
        associationRules = self.sortAssociationRules(associationRules) # sorts rules by length, lift, confidence and support
        print("Complete.")
        return frequentSets, associationRules

    def generateAssociationRules(self, frequentSets:list)->list:
        """
        Generates a list of association rules that meet the confidence and lift thresholds

        Parameters:
        frequentSets (list): A list of frequent itemsets

        Returns:
        list: A list containing the association rules that pass the support, confidence and lift thresholds
        """
        associationRules = []
        # partition the set  
        for itemset in frequentSets:
            for i in range(len(itemset)):
                for c in [c for c in itertools.combinations(itemset, i+1)]:
                    body = set(list(c))
                    head = set([i for i in itemset if not i in c])
                    if (len(head) > 0 and len(body) > 0):
                        # from head/body partitions create association rules 
                        associationRule = AssociationRule(body, head) # X -> Y
                        associationRule.support = self.calculateSupport(associationRule.itemset)
                        associationRule.confidence = self.calculateConfidence(associationRule.itemset, associationRule.body)
                        associationRule.lift = self.calculateLift(associationRule.body, associationRule.head, associationRule.confidence)
                        # test is rule passes minconf, minsup and minlift
                        if (associationRule.confidence >= self.minconf) and (associationRule.support >= self.minsup) and (associationRule.lift >= self.minlift):
                            associationRules.append(associationRule)
        return associationRules

    def generateFrequentSets(self)->list:
        """
        Implements the Apriori frequent itemset generation algorithm.
        Where itemsets is c_k and frequentSets = l_k 

        Returns:
        list: A list of frequent itemsets of size k-1
        """
        allFrequentSets = []

        # Generate frequent itemsets of length k
        frequentSets, infrequentSets = self.eliminateCandidates(self.items) # c_1
        for fset in frequentSets:
            allFrequentSets.append(fset)
        # Repeat until no new frequent itemsets are identified
        k = 1
        while True:  
            k += 1
            print("     k = " + str(k) + "    " , end="\n")
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
            # stores all the frequent itemsets of size k
            for fset in frequentSets:
                allFrequentSets.append(fset)
        return allFrequentSets

    def sortAssociationRules(self, associationRules:list)->list: # TODO: sorted output list
        """
        Sorts the given list of association rules by four different attributes.
        Including by itemset length (decreasing), minlift, minconf and minsup (decreasing)

        Paramaters:
        associationRules (list): A list of AssociationRule objects

        Returns:
        list: A sorted list of AssociationRule objects

        """
        associationRules = sorted(associationRules, key=lambda x: x.support, reverse=True) # support (decreasing)
        associationRules = sorted(associationRules, key=lambda x: x.confidence, reverse=False) # confidence
        associationRules = sorted(associationRules, key=lambda x: x.lift, reverse=False) # lift value
        associationRules = sorted(associationRules, key=lambda x: len(x.itemset), reverse=True) # number of items (decreasing)

        return associationRules
    
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

    def calculateConfidence(self, itemset:set, body:set)->float:
        """
        Calculates how often items in Y appear in transactions containing X

        Parameters
        set (set): A set of items

        Returns
        float: The confidence value for that set (rule)

        """
        return self.calculateSupport(itemset) / self.calculateSupport(body)

    def calculateLift(self, body:list, head:list, confidence:float)->float:
        """
        Calculates the importance of a rule
        """
        bodySupport = self.calculateSupport(body)
        headSupport = self.calculateSupport(head)
        return confidence / ((bodySupport * headSupport) / bodySupport)

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
            for i in range(len(itemsets)):
                #print("     " +str(round(i / len(itemsets) * 100, 2)) + "%", end="")
                #print("\r", end="")
                for j in range(len(infrequentSets)):
                    if itemsets[i] == None: 
                        break # set has been pruned so all other comparisons can be skipped
                    elif (infrequentSets[j].issubset(itemsets[i])):
                        itemsets[i] = None
            prunedSets = [itemset for itemset in itemsets if itemset is not None] # filter all nonpruned itemsets
            return prunedSets

def displayAssociationRules(associationRules:list):
    """
    Displays each association rule in a df alongside itemset frequency, 
    lift, confidence and support values.

    Parameters:
    associationRules (list): A list of sorted association rules
    """
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    ruleList = []
    for associationRule in associationRules:
        rule = [str(associationRule), len(associationRule.itemset), round(associationRule.lift, 2), associationRule.confidence, associationRule.support]
        ruleList.append(rule)
    df = pd.DataFrame(ruleList, columns=["Rule", "Length", "Lift", "Conf", "Sup"])
    print(df.to_string(index=False))

def main():
    startTime = time.time()

    path = 'supermarket.csv'
    apriori = Apriori(minsup=0.15, minconf=0.8, minlift=0, path=path)
    frequentSets, associationRules = apriori.run()

    print("\nComputed association rules:\n")
    displayAssociationRules(associationRules)

    print("\nFound {} rules from {} itemsets in {} seconds.\n".format(len(associationRules), len(frequentSets), round(time.time() - startTime, 2)))

if __name__ == "__main__":
    main()
