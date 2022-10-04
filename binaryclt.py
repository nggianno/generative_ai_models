import random
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import csv
import pandas as pd
import math



class BinaryCLT:
    def __init__(self, data, root: int = 0, alpha: float = 0.01):

        self.root = root
        self.data = data
        self.alpha = alpha

        if root == None:
            root = np.random.choice(data.shape[1])

        # calculate mutual information for all RV combinations and fill the table
        mi_table = np.zeros((data.shape[1], data.shape[1]))

        for xi in range(data.shape[1]):
            for xj in range(xi + 1, data.shape[1]):
                mi_val = self.mutual_info(data, xi, xj, alpha)
                mi_table[xi, xj] = mi_val
                # mi_table[xj, xi] = mi_val

        # calculate the minimum spanning tree
        max_span_tree = minimum_spanning_tree(-mi_table)
        max_span_tree = max_span_tree.toarray()

        # get directed tree
        directed_tree = breadth_first_order(max_span_tree, 0, directed=False, return_predecessors=True)
        print(directed_tree)
        self.directed_tree = directed_tree

    def joint_prob(self, df, alpha, rv1, rv2, val1, val2):

        joint_p = (alpha + len(df[(df[rv1] == val1) & (df[rv2] == val2)])) / (4 * alpha + len(df))

        return joint_p

    def single_prob(self,df, alpha, rv, val):

        single_p = (2 * alpha + len(df[(df[rv] == val)])) / (4 * alpha + len(df))

        return single_p

    def mutual_info(self, df, rv1, rv2, alpha):

        summ = 0
        for i in [0, 1]:
            for j in [0, 1]:
                summ += self.joint_prob(df, alpha, rv1, rv2, i, j) * \
                        math.log(
                            self.joint_prob(df, alpha, rv1, rv2, i, j) /(
                            self.single_prob(df, alpha, rv1, i) * self.single_prob(df,alpha,rv2,j)))

        return summ


    #2a
    def get_tree(self):

        tree = self.directed_tree[1]
        tree = [-1 if item == -9999 else item for item in tree]

        return tree

    #2b
    def get_log_params(self):

        log_params = np.zeros((self.data.shape[1], 2, 2))
        tree = self.get_tree()

        for i in range(df.shape[1]):

            # if (tree[i] == -9999) and (i != self.root):
            #     continue
            # else:

            if i == self.root:
                log_params[i, 0, 0] = math.log(self.single_prob(df, self.alpha, i, 0))
                log_params[i, 0, 1] = math.log(self.single_prob(df, self.alpha, i, 1))
                log_params[i, 1, 0] = log_params[i, 0, 0]
                log_params[i, 1, 1] = log_params[i, 0, 1]


            else:
                # print(i, tree[i])
                for j in [0, 1]:
                    for k in [0, 1]:
                        cond_p = self.joint_prob(df,self.alpha, i, tree[i], j, k) / self.single_prob(df, self.alpha, tree[i],k)
                        log_params[i, j, k] = math.log(cond_p)

        return log_params

    #2c
    def log_prob(self, x, exhaustive: bool = False):

        pass


    #2d
    def sample(self, n_samples: int):

        samples = []
        tree_order_nodes = self.directed_tree[0]
        node_parents = self.directed_tree[1]

        for _ in range(n_samples):
            new_sample_list = np.zeros(self.data.shape[1])
            for node in tree_order_nodes:
                random_num = random.random()
                log_parameters = self.get_log_params()
                probs = np.exp(log_parameters)

                if node == 0:
                    if random_num <= probs[node][1][1]:
                        new_sample_list[node] = 1
                    else:
                        new_sample_list[node] = 0
                else:
                    val = int(new_sample_list[node_parents[node]])
                    if random_num <= probs[node][1][val]:
                        new_sample_list[node] = 1
                    else:
                        new_sample_list[node] = 0
                sample_list = list(new_sample_list)
                sample_list = [int(x) for x in sample_list]

            samples.append(sample_list)

        return samples


if __name__ == "__main__":

    # read dataset
    with open("binary_datasets/nltcs/nltcs.train.data", "r") as file:
        reader = csv.reader(file, delimiter=',')
        dataset = np.array(list(reader)).astype(np.float)

    df = pd.DataFrame(dataset, columns=list(range(0, dataset.shape[1])))

    """Call the constructor of BinaryCLT"""
    clt = BinaryCLT(data = df)

    """Display the Chow-Liu Tree"""
    tree_res = clt.get_tree()
    print("Chow-Liu Tree", tree_res)

    """Display the log parameters"""
    log_parameters = clt.get_log_params()
    probabilities = np.exp(log_parameters)
    # print("CPT Log Parameters", log_parameters)
    # print("CPT Parameters", probabilities)

    """Generate samples with Ancestral Sampling technique"""
    samples = clt.sample(n_samples = 10)





