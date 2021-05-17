class AssociationRule():
    """
    Represents an association rule in the form {body} -> {head}

    Parameters:
    body (set): The set of items in the body of the association rule
    head (set): The set of items in the head of the association rule

    """
    def __init__(self, body:set, head:set):
        self.head = head
        self.body = body
        self.itemset = body.union(head)
        self.confidence = None
        self.support = None
        self.lift = None

    def __str__(self) -> str:
        return "{} -> {}".format(self.body, self.head)