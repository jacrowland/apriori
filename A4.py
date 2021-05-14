import numpy as np
import pandas as pd

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
        items (set): The set of unique items found across all transactions - Initally empty

        Returns:
        None:Returning value

    """
        self.minsup = minsup
        self.minconf = minconf
        self.minlift = minlift
        self.path = path
        self.items = set()

    def importTransactions():
        pass



    def calculateSupport(set:set)->float:
        """
        Calculates the support value of a set by dividing the number of transactions a set occurs in with the total number of transactions

        Parameters
        set (set): A set of items

        Returns:
        float: The support value for that set (rule)

        """
        pass

    def calculateConfidence(set:set)->float:
        """
        Calculates how often items in Y appear in transactions containing X

        Parameters
        set (set): A set of items

        Returns:
        float: The confidence value for that set (rule)

        """
        pass

def main():
    pass

if __name__ == "__main__":
    main()