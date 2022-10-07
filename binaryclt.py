import random
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import csv
import pandas as pd
import math
import itertools
import time


class BinaryCLT:
    def __init__(self, data, root: int = 0, alpha: float = 0.01):

        #Initialized variables from constructor
        self.root = root
        self.data = data
        self.alpha = alpha

        print("Root:", root)
        print("Alpha:", alpha)

        #if given root index is None, then randomly pick a RV as root
        if root == None:
            self.root = np.random.choice(data.shape[1])


        data = pd.DataFrame(data, columns=list(range(0, data.shape[1])))
        df = data
        self.df = df

        # calculate mutual information for all RV combinations and fill the table
        mi_table = np.zeros((data.shape[1], data.shape[1]))

        for xi in range(data.shape[1]):
            for xj in range(xi + 1, data.shape[1]):
                mi_val = self.mutual_info(data, xi, xj, alpha)
                mi_table[xi, xj] = mi_val

        # calculate the maximum spanning tree
        max_span_tree = minimum_spanning_tree(-mi_table)
        max_span_tree = max_span_tree.toarray()

        # get directed tree
        directed_tree = breadth_first_order(max_span_tree, 0, directed=False, return_predecessors=True)
        print(directed_tree)
        self.directed_tree = directed_tree

    #calculation of joint probability
    def joint_prob(self, df, alpha, rv1, rv2, val1, val2):

        joint_p = (alpha + len(df[(df[rv1] == val1) & (df[rv2] == val2)])) / (4 * alpha + len(df))

        return joint_p

    #calculation of single probability
    def single_prob(self,df, alpha, rv, val):

        single_p = (2 * alpha + len(df[(df[rv] == val)])) / (4 * alpha + len(df))

        return single_p

    #calculation of mutual information
    def mutual_info(self, df, rv1, rv2, alpha):

        summ = 0
        for i in [0, 1]:
            for j in [0, 1]:
                summ += self.joint_prob(df, alpha, rv1, rv2, i, j) * \
                        math.log(
                            self.joint_prob(df, alpha, rv1, rv2, i, j) /(
                            self.single_prob(df, alpha, rv1, i) * self.single_prob(df,alpha,rv2,j)))

        return summ


    #2a - get the Chow Liu Tree
    def get_tree(self):

        tree = self.directed_tree[1]
        tree = [-1 if item == -9999 else item for item in tree]

        return tree

    #2b - calculate the log parameters table
    def get_log_params(self):

        log_params = np.zeros((self.data.shape[1], 2, 2))
        tree = self.get_tree()

        for i in range(self.df.shape[1]):


            if i == self.root:
                log_params[i, 0, 0] = math.log(self.single_prob(self.df, self.alpha, i, 0))
                log_params[i, 0, 1] = math.log(self.single_prob(self.df, self.alpha, i, 1))
                log_params[i, 1, 0] = log_params[i, 0, 0]
                log_params[i, 1, 1] = log_params[i, 0, 1]


            else:
                for j in [0, 1]:
                    for k in [0, 1]:
                        cond_p = self.joint_prob(self.df,self.alpha, i, tree[i], j, k) / self.single_prob(self.df, self.alpha, tree[i],k)
                        log_params[i, j, k] = math.log(cond_p)

        return log_params

    #2c - calculate the log probabilities for each sample with exhaustive and non-exhaustive inference
    def log_prob(self, x, exhaustive: bool = False):

        lp = []
        samples_nums = x.shape[0]
        log_params = self.get_log_params()
        tree = self.get_tree()

        if exhaustive:

            """Exhaustive Approach"""

            for sample_idx in range(samples_nums):
                query = x[sample_idx]
                nan_num = sum(np.isnan(query))
                nan_indices = np.argwhere(np.isnan(query))
                lst = list(itertools.product([0, 1], repeat=nan_num))

                combinations = []
                query_dupl = query.copy()

                for i in range(2 ** nan_num):
                    for j, index in enumerate(nan_indices):

                        query_dupl[index[0]] = lst[i][j]

                    data_to_add = query_dupl.tolist()
                    combinations.append(data_to_add)


                logsumexp_inputs = []
                for comb in combinations:
                    summ = 0
                    for rv_idx in range(len(log_params)):
                        val_idx = int(comb[rv_idx])
                        if rv_idx == 0:
                            summ += log_params[rv_idx][0][val_idx]
                        else:
                            parent_idx = int(comb[tree[rv_idx]])
                            summ += log_params[rv_idx][parent_idx][val_idx]
                    logsumexp_inputs.append(summ)
                res = logsumexp(logsumexp_inputs)

                lp.append(res)

        else:

            """Non-exhaustive Approach | Message Passing """
            root = self.root
            tree_arr = np.array(tree)


            def get_log_probability(sample_query, tree):

                children = np.argwhere(tree==0).flatten()
                root_unknown = np.isnan(sample_query[root])
                if not root_unknown:
                    parent_val = int(sample_query[root])
                    prob_x = log_params[root, 0, parent_val] + \
                           sum([get_children_node_values(sample_query, tree, child_idx, parent_val) for child_idx in children])
                    return prob_x
                else:
                    terms_sum = [log_params[root, 0, binary_root_val] +
                             sum([get_children_node_values(sample_query, tree, child_idx, binary_root_val) for child_idx in children])
                             for binary_root_val in [0, 1]]
                    return logsumexp(terms_sum)

            def get_children_node_values(sample_query, tree, child_idx, parent_val):
                children_of_child = np.argwhere(tree == child_idx).flatten()
                child_unknown = np.isnan(sample_query[child_idx])
                if len(children_of_child) > 0:
                    if not child_unknown:
                        prob_x = log_params[child_idx, parent_val, int(sample_query[child_idx])] + \
                               sum([get_children_node_values(sample_query, tree, child, int(sample_query[child_idx]))
                                    for child in children_of_child])
                        return prob_x
                    else:
                        terms_sum = [log_params[child_idx, parent_val, binary_root_val] +
                                 sum([get_children_node_values (sample_query, tree, child, binary_root_val)
                                      for child in children_of_child])
                                 for binary_root_val in [0, 1]]
                        return logsumexp(terms_sum)
                else:
                    if not child_unknown:
                        return log_params[child_idx, parent_val, int(sample_query[child_idx])]
                    else:
                        terms_sum = [sum([log_params[child_idx, parent_val, value]])
                                 for value in [0, 1]]
                        return logsumexp(terms_sum)

            for sample_idx in range(samples_nums):

                query = x[sample_idx]
                log_prob_sample_val = get_log_probability(query, tree_arr)
                lp.append(log_prob_sample_val)

        return lp


    #2d - Ancestral Sampling
    def sample(self, n_samples: int):

        samples = []
        tree_order_nodes = self.directed_tree[0]
        node_parents = self.directed_tree[1]
        log_parameters = self.get_log_params()
        probs = np.exp(log_parameters)


        for _ in range(n_samples):

            new_sample_list = np.zeros(self.data.shape[1])
            for node in tree_order_nodes:
                random_num = random.random()

                if node == 0:
                    if random_num <= probs[node][1][1]:
                        new_sample_list[node] = 1
                    else:
                        new_sample_list[node] = 0
                else:
                    val = int(new_sample_list[node_parents[node]])
                    if random_num <= probs[node][val][1]:
                        new_sample_list[node] = 1
                    else:
                        new_sample_list[node] = 0
                sample_list = list(new_sample_list)
                sample_list = [int(x) for x in sample_list]

            samples.append(sample_list)

        return np.array(samples)


if __name__ == "__main__":

    # read dataset
    with open("binary_datasets/nltcs/nltcs.train.data", "r") as file:
        reader = csv.reader(file, delimiter=',')
        train_data = np.array(list(reader)).astype(np.float)

    with open("binary_datasets/nltcs/nltcs.test.data", "r") as file:
        reader = csv.reader(file, delimiter=',')
        test_data = np.array(list(reader)).astype(np.float)

    with open("nltcs_marginals.data", "r") as file:
        reader = csv.reader(file, delimiter=',')
        marginals_data = np.array(list(reader)).astype(np.float)


    """Call the constructor of BinaryCLT"""
    clt = BinaryCLT(data = train_data)
    #
    """Display the Chow-Liu Tree"""
    tree_res = clt.get_tree()
    print("Chow-Liu Tree", tree_res)
    #
    """Display the log parameters"""
    log_parameters = clt.get_log_params()
    probabilities = np.exp(log_parameters)
    print("CPT Log Parameters", log_parameters)
    print("CPT Parameters", probabilities)

    """Get average probability of train and test data"""
    # probs_train = clt.log_prob(x = train_data, exhaustive= True)
    # probs_test = clt.log_prob(x = test_data, exhaustive= True)
    # # print("Probs Train shape:",np.array(probs_train).shape)
    # # print("Probs Test shape:",np.array(probs_test).shape)
    # avg_train =  sum(probs_train) / len(probs_train)
    # avg_test =  sum(probs_test) / len(probs_test)
    #
    # # avg_train = np.mean(probs_train)
    # # avg_test = np.mean(probs_test)
    #
    # print("Average Train Probability",avg_train)
    # print("Average Test Probability",avg_test)

    """Generate samples with Ancestral Sampling technique"""
    # samples = clt.sample(n_samples = 1000)
    # print(samples)
    #
    # samples_data = np.array(samples)
    # # print(samples_data.shape)
    # # samples_data_df = pd.DataFrame(train_data, columns=list(range(0, samples_data.shape[1])))
    # lp_samples_data = clt.log_prob(x=samples_data, exhaustive=True)
    # avg_samples_data = sum(lp_samples_data) / len(lp_samples_data)
    # print(avg_samples_data)

    # clt2 = BinaryCLT(data = samples_data_df)
    # tree_res = clt2.get_tree()
    # print("Chow-Liu Tree", tree_res)
    # log_parameters = clt2.get_log_params()
    # probabilities = np.exp(log_parameters)

    # x_data = [[1, 0, 1, 1, 0], [np.nan, 0, np.nan, np.nan, 1], [np.nan, 0, 1, 1, np.nan]]
    # x_data = np.array(x_data)

    # start1 = time.time()
    # print("Started timing exhaustive...")
    # lp_exhaustive = clt.log_prob(x = marginals_data, exhaustive= True)
    # print("Stopped timing exhaustive...")
    # end1 = time.time()
    # print(end1 - start1)
    # # lp_exhaustive_rounded = [round(num, 5) for num in lp_exhaustive]
    # # print(lp_exhaustive_rounded)
    #
    # start2 = time.time()
    # print("Started timing non-exhaustive...")
    # lp_non_exhaustive = clt.log_prob(x = marginals_data, exhaustive= False)
    # print("Stopped timing non-exhaustive...")
    # end2= time.time()
    # print(end2 - start2)
    # # lp_non_exhaustive_rounded = [round(num, 5) for num in lp_non_exhaustive]
    # # print(lp_non_exhaustive_rounded)
    #
    # diff_lp = lp_exhaustive - lp_non_exhaustive
    # avg_diff_lp = sum(diff_lp) / len(diff_lp)
    # print(avg_diff_lp)





