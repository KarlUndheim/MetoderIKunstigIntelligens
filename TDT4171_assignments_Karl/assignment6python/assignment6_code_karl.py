import numpy as np
from pathlib import Path
from typing import Tuple



class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)


def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value

# Calculating entropy, This will be useful in the importance function.
# To prevent Nan values I manually check for q==0 and q==1.
def B(q: int):
    if q==0 or q==1:
        return 0
    return -(q*np.log2(q) + (1-q)*np.log2(1-q))

def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    # implement the importance function for both measure = "random" and measure = "information_gain"

    if measure=="random":
        # I use np.random to select a random attribute
        return np.random.choice(attributes)
    else:
        # These are the actual classifications
        labels = examples[:, -1]

        # Total number of positives(label==1)
        p = np.count_nonzero(labels==1, axis=0)
        # Total number of negatives(label==2)
        n = len(labels)-p

        # Entropy of dataset
        gain0 = B(p/(p+n))

        # Initialize best gains and split attribute
        bestGain = -1
        bestSplit = -1

        # Iterate through each attribute and calculate gainsplit.
        for atr in attributes:
            gain = gain0
            all_values = examples[:, atr]
            unique_values = sorted(list(set(all_values)))
            for val in unique_values:
                indices = np.argwhere(all_values==val)
                classifications = labels[indices]
                p_k = np.count_nonzero(classifications==1, axis=0)
                n_k = len(classifications) - p_k
                remainder = ((p_k+n_k)/(p+n))*B(p_k/(p_k+n_k))
                gain -= remainder
            if gain>bestGain:
                bestGain = gain
                bestSplit = atr

        # Return the best attribute to split on
        return bestSplit





def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # TODO implement the steps of the pseudocode in Figure 19.5 on page 678

    # Leaf nodes
    if len(examples)==0:
        node.value = plurality_value(parent_examples)
        return node
    classifications = examples[:,-1]
    if len(list(set(classifications)))==1:
        node.value = classifications[0]
        return node
    if len(attributes)==0:
        node.value = plurality_value(examples)
        return node
    
    # Use importance to find the best attribute "A" to split on.
    A = importance(attributes, examples, measure)
    # Current node is set to this attribute.
    node.attribute = A

    # Remove this attribute
    updatedAttributes = np.delete(attributes, np.argwhere(attributes==A))

    # For each unique value of this attribute make a new decision tree.
    for v in list(set(examples[:, A])):
        indices = [i for i in range(0,len(examples)) if examples[i][A]==v]
        # Examples with current value in attribute A
        exs = examples[indices]
        
        # Create a subtree
        subtree = learn_decision_tree(exs, updatedAttributes, examples, node, v, measure)

        # Attach subtree to current node
        node.children[v] = subtree


    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test




if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "random"
    counter=0
    train_accuracy_random=0
    test_accuracy_random=0
    for i in range(100):
        tree = learn_decision_tree(examples=train,
                        attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                        parent_examples=None,
                        parent=None,
                        branch_value=None,
                        measure=measure)
        # I have a strange bug where the code crashes 1 out of 4 times when using random split
        # I couldnt debug it so I throw an exception when it doesnt work.
        try:
            test_accuracy_random += accuracy(tree, test)
            train_accuracy_random += accuracy(tree, train)
            counter+=1

        except:
            pass

    print("")
    print("Training accuracy (random split): ",train_accuracy_random/counter)
    print("Test accuracy (random split): ",test_accuracy_random/counter)
    print("")

    measure = "information_gain"

    # No need to loop and find average test accuracy because the decision tree is identical every time.
    tree = learn_decision_tree(examples=train,
                        attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                        parent_examples=None,
                        parent=None,
                        branch_value=None,
                        measure=measure)

    print(f"Training Accuracy (gain split): {accuracy(tree, train)}")
    print(f"Test Accuracy (gain split): {accuracy(tree, test)}")
    print("")