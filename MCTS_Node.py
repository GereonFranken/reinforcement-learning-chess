import numpy as np

class Node:
    """
    Node for the MCTS. Stores the move applied to reach this node from its parent,
    stats for the associated game position, children, parent and outcome
    (outcome==none unless the position ends the game).
    Args:
        move:
        parent:
        N (int): times this position was visited.
        Q (int): average reward (wins-losses) from this position.
        Q_RAVE (int): will be explained later.
        N_RAVE (int): will be explained later.
        children (dict): dictionary of successive nodes.
        outcome (int): If node is a leaf, then outcome indicates
                       the winner, else None.
    """

    def __init__(self, move=None, parent=None):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.
        """
        self.move = move
        self.parent = parent
        self.N = 0  # times this position was visited
        self.Q = 0  # average reward (wins-losses) from this position
        self.N_RAVE = 0
        self.Q_RAVE = 0
        self.children = {}

    def add_children(self, children):
        """
        Add a list of nodes to the children of this node.
        """
        for child in children:
            self.children[child.move] = child

    @property
    def value(self, explore=0.5):
        """
        Calculate the UCT value of this node relative to its parent, the parameter
        "explore" specifies how much the value should favor nodes that have
        yet to be thoroughly explored versus nodes that seem to have a high win
        rate.
        Currently explore is set to 0.5.
        """
        # if the node is not visited, set the value as infinity. Nodes with no visits are on priority
        if self.N == 0:
            return 0 if explore == 0 else np.inf
        else:
            return self.Q / self.N + explore * np.sqrt(2 * np.log(self.parent.N) / self.N)  # exploitation + exploration

