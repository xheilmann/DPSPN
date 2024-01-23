
import numpy as np

from diffprivlib.mechanisms import GeometricTruncated
from diffprivlib.accountant import BudgetAccountant

from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.Validity import is_valid
from spn.algorithms.splitting.Random import get_split_cols_random_partition, get_split_cols_binary_random_partition
from spn.structure.Base import Sum, assign_ids
from sys import maxsize

from dpspn.dp_row_splitting import get_split_rows_dp_KMeans
from dpspn.dp_Histogram_leaves import create_dp_histogram_leaf

import logging

from dpspn.dp_structure_learning import learn_dp_structure, get_next_operation

logger = logging.getLogger(__name__)

def find_max_index(lst):
    if not lst:
        return None
    max_index = 0
    max_element = lst[0].total()

    for i in range(1, len(lst)):
        if lst[i].total() > max_element:
            max_element = lst[i].total()
            max_index = i

    return max_index

def get_splitting_functions( cols, rows, epsilon, range, accountant, threshold):

    if isinstance(cols, str):
        rand_gen=np.random.RandomState()
        if cols == "binary_random":

            split_cols = get_split_cols_binary_random_partition(threshold=threshold, rand_gen =rand_gen)
        elif cols == "random":
            split_cols = get_split_cols_random_partition(rand_gen, ohe=False)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):

        if rows == "kmeans":
            split_rows = get_split_rows_dp_KMeans(epsilon=epsilon,  bounds= range, accountant=accountant)

        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows

    return split_cols,  split_rows

def learn_dp_classifier(data, ds_context, spn_learn_wrapper, label_idx, epsilon,**kwargs):
    spn = Sum()
    dp_mech = GeometricTruncated(epsilon=epsilon, sensitivity=1, lower=0, upper=maxsize)
    max_op = []

    accountants=[]
    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        acc = BudgetAccountant()
        #print(label)
        branch, max_op_on_dataset_slice = spn_learn_wrapper(data[data[:, label_idx] == label, :], ds_context, epsilon=epsilon,**kwargs)
        #print(get_structure_stats(branch))
        spn.children.append(branch)

        count = np.uint32(dp_mech.randomise(count))
        spn.weights.append(count / data.shape[0])
        max_op.append(max_op_on_dataset_slice)
        acc.spend(epsilon=(max_op_on_dataset_slice*epsilon),delta= 0)
        print(acc.total())
        accountants.append(acc)

    weightsum = np.sum(spn.weights)
    for i in range(len(spn.weights)):
        spn.weights[i] =spn.weights[i] / weightsum

    final_accountant=accountants[find_max_index(accountants)]
    #print(final_accountant.total())

    spn.scope.extend(branch.scope)
    assign_ids(spn)

    valid, err = is_valid(spn)
    assert valid, "invalid spn: " + err
    final_accountant.spend(epsilon, 0)
    #print(final_accountant.total())

    return spn, max(max_op) +1


def learn_dp_mspn(
    data,
    ds_context,
    rows="kmeans",
    cols=None,
    min_instances_slice=200,
    leaves=None,
    epsilon = 1.0,
    range = None,
    accountant = None,
    total_operations=10,
    threshold=0.3
):

    if leaves is None:
        leaves = create_dp_histogram_leaf
    print(total_operations)


    def l_dp_mspn(data, ds_context, cols, rows, min_instances_slice, leaves, epsilon, range, accountant, total_operations, threshold):
        split_cols, split_rows = get_splitting_functions( cols, rows, epsilon, range, accountant, threshold)

        nextop = get_next_operation(min_instances_slice)

        return learn_dp_structure(data, ds_context, split_cols, split_rows, leaves,epsilon,range, accountant, nextop,  total_operations=total_operations)
    #
    #accountant.spend(epsilon, 0)
    return l_dp_mspn(data, ds_context,cols,  rows, min_instances_slice, leaves,  epsilon, range, accountant, total_operations, threshold)







