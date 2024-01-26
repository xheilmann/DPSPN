"""
Code inspired by the SPFlow library (https://github.com/SPFlow/SPFlow)
"""
import itertools
import logging
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

from spn.algorithms.TransformStructure import Prune
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, assign_ids
import multiprocessing
import os
from diffprivlib.tools.utils import var as dpvar
from dpspn.dp_row_splitting import get_bounds

parallel = True

if parallel:
    cpus = max(1, os.cpu_count() - 2)  # - int(os.getloadavg()[2])
else:
    cpus = 1
pool = multiprocessing.Pool(processes=cpus)

#test
class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4
    REMOVE_UNINFORMATIVE_FEATURES = 5
    CONDITIONING = 6


def get_next_operation(min_instances_slice=100, min_features_slice=1, multivariate_leaf=False, cluster_univariate=False, accountant=None):
    def next_operation(
        data,
        scope,
        create_leaf,
        no_clusters=False,
        no_independencies=False,
        is_first=False,
        cluster_first=True,
        operations_counter = 0,
        total_operations=20,
        epsilon=1.0,
        bounds = None,
        accountant=None,
    ):

        minimalInstances = data.shape[0] <= min_instances_slice
        if operations_counter>=total_operations-2 or minimalInstances:
            return Operation.NAIVE_FACTORIZATION, True


        uninformative_features_idx = dpvar(data[:, 0 : len(scope)], epsilon=epsilon, bounds = bounds, accountant=accountant) == 0
        #print(accountant.total())

        ncols_zero_variance = np.sum(uninformative_features_idx)
        #ncols_zero_variance =0
        if ncols_zero_variance > 0 and (operations_counter < total_operations -1):

            if ncols_zero_variance == data.shape[1]:
                if multivariate_leaf:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            else:
                return (
                    Operation.REMOVE_UNINFORMATIVE_FEATURES,
                    np.arange(len(scope))[uninformative_features_idx].tolist(),
                )

        if minimalInstances or (no_clusters and no_independencies) or (operations_counter>=total_operations-1):
            if multivariate_leaf:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None
        if operations_counter < total_operations -1:

            return Operation.SPLIT_ROWS, None





        return Operation.NAIVE_FACTORIZATION, None

    return next_operation


def default_slicer(data, cols, num_cond_cols=None):
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))

        return data[:, cols]
    else:
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)


def inList(array, list):
    for k in list:
        if np.array_equal(array,k) == True:
            return False
        else:
            continue
    return True



def learn_dp_structure(
    dataset,
    ds_context,
    split_cols,
    split_rows,
    create_leaf,
    epsilon,
    bounds,
    accountant,
    next_operation=get_next_operation(),
    total_operations = 20,
    initial_scope=None,
    data_slicer=default_slicer

):
    assert dataset is not None
    assert ds_context is not None
    assert split_rows is not None
    assert create_leaf is not None
    assert next_operation is not None

    #print(split_cols)
    #print(split_rows)


    counter = {"col": 0, "row": 0, "ruf":0,"leaf": [0 for i in range((dataset.shape[1]))]}
    dataset_slices = {"col": [], "row": {}, "leaf": {i: [] for i in range((dataset.shape[1]))}}


    root = Product()
    root.children.append(None)


    if initial_scope is None:
        initial_scope = list(range(dataset.shape[1]))
        num_conditional_cols = None
    elif len(initial_scope) < dataset.shape[1]:
        num_conditional_cols = dataset.shape[1] - len(initial_scope)
    else:
        num_conditional_cols = None
        assert len(initial_scope) > dataset.shape[1], "check initial scope: %s" % initial_scope

    tasks = deque()
    tasks.append((dataset, root, 0, initial_scope, False, False, True, 0, total_operations))

    while tasks:


        local_data, parent, children_pos, scope, no_clusters, no_independencies, cluster_first, operations_counter, total_operations  = tasks.popleft()
        bounds = get_bounds(ds_context, scope)

        operation, op_params = next_operation(
            local_data,
            scope,
            create_leaf,
            no_clusters=no_clusters,
            no_independencies=no_independencies,
            is_first=(parent is root),
            cluster_first=cluster_first,
            operations_counter = operations_counter,
            total_operations=total_operations,
            epsilon = epsilon,
            bounds = bounds,
            accountant = accountant,
        )

        logging.debug("OP: {} on slice {} (remaining tasks {})".format(operation, local_data.shape, len(tasks)))

        if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:


            if f"({local_data})" in dataset_slices["row"]:
                dataset_slices["row"][f"({local_data})"] +=1
            else:
                dataset_slices["row"][f"({local_data})"] = 1

            operations_counter = dataset_slices["row"].get(f"({local_data})")

            counter["ruf"] +=1
            node = Product()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            rest_scope = set(range(len(scope)))
            for col in op_params:

                rest_scope.remove(col)
                node.children.append(None)
                tasks.append(
                    (
                        data_slicer(local_data, [col], num_conditional_cols),
                        node,
                        len(node.children) - 1,
                        [scope[col]],
                        True,
                        True,
                        True,
                        operations_counter,
                        total_operations

                    )
                )

            next_final = False

            if len(rest_scope) == 0:
                continue
            elif len(rest_scope) == 1:
                next_final = True

            node.children.append(None)
            c_pos = len(node.children) - 1

            rest_cols = list(rest_scope)
            rest_scope = [scope[col] for col in rest_scope]

            tasks.append(
                (
                    data_slicer(local_data, rest_cols, num_conditional_cols),
                    node,
                    c_pos,
                    rest_scope,
                    next_final,
                    next_final,
                    True,
                    operations_counter,
                    total_operations
                )
            )

            continue

        elif operation == Operation.SPLIT_ROWS:
            if not op_params:
                if f"({local_data})" in dataset_slices["row"].keys():
                    dataset_slices["row"][f"({local_data})"] +=2
                else:
                    dataset_slices["row"][f"({local_data})"] = max(operations_counter,2)
            else:
                if f"({local_data})" in dataset_slices["row"].keys():
                    dataset_slices["row"][f"({local_data})"] += 1
                else:
                    dataset_slices["row"][f"({local_data})"] = max(operations_counter,1)
            operations_counter = dataset_slices["row"].get(f"({local_data})")

            split_start_t = perf_counter()
            data_slices = split_rows(local_data, ds_context, scope)
            split_end_t = perf_counter()
            logging.debug(
                "\t\tDP found {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
            )
            if len(data_slices) !=1:
                counter["row"]+=1

            if len(data_slices) == 1:


                tasks.append((local_data, parent, children_pos, scope, True, False, True, operations_counter, total_operations))
                continue

            node = Sum()
            node.scope.extend(scope)
            parent.children[children_pos] = node
            # assert parent.scope == node.scope



            for data_slice, scope_slice, proportion in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"
                #operations_counter = 0
                #print(operations_counter)

                node.weights.append(proportion)
                if split_cols is None:
                    node.children.append(None)
                    tasks.append((data_slice, node, len(node.children) - 1, scope, False, False, True, operations_counter, total_operations))
                if len(data_slices) != 1 and len(scope_slice)>1 and split_cols is not None:
                    split_start_t = perf_counter()
                    col_data_slices = split_cols(data_slice, ds_context, scope)
                    split_end_t = perf_counter()
                    logging.debug(
                        "\t\tDP found {} col clusters (in {:.5f} secs)".format(len(col_data_slices),
                                                                               split_end_t - split_start_t)
                    )
                    if len(col_data_slices) == 1:
                        node.children.append(None)
                        tasks.append((data_slice, node, len(node.children) - 1, scope, False, False, True,
                                      operations_counter, total_operations))
                    else:
                        col_node = Product()
                        col_node.scope.extend(scope)
                        node.children.append(col_node)

                        for col_data_slice, col_scope_slice, _ in col_data_slices:
                            assert isinstance(col_scope_slice, list), "slice must be a list"

                            col_node.children.append(None)
                            tasks.append((col_data_slice, col_node, len(col_node.children) - 1, col_scope_slice, False, False, True, operations_counter, total_operations))



            continue



        elif operation == Operation.NAIVE_FACTORIZATION:

            if f"({local_data})" in dataset_slices["row"]:
                dataset_slices["row"][f"({local_data})"] +=1
            else:
                dataset_slices["row"][f"({local_data})"] = max(operations_counter,1)





            node = Product()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            local_tasks = []
            local_children_params = []
            split_start_t = perf_counter()
            for col in range(len(scope)):
                node.children.append(None)
                # tasks.append((data_slicer(local_data, [col], num_conditional_cols), node, len(node.children) - 1, [scope[col]], True, True))
                local_tasks.append(len(node.children) - 1)
                child_data_slice = data_slicer(local_data, [col], num_conditional_cols)
                local_children_params.append((child_data_slice, ds_context, [scope[col]], epsilon, bounds, accountant))
                counter["leaf"][col] += 1
                dataset_slices["leaf"][col].append(child_data_slice)

            result_nodes = pool.starmap(create_leaf, local_children_params)
            #accountant.spend(epsilon=epsilon, delta=0)
            # result_nodes = []
            # for l in tqdm(local_children_params):
            #    result_nodes.append(create_leaf(*l))
            # result_nodes = [create_leaf(*l) for l in local_children_params]
            for child_pos, child in zip(local_tasks, result_nodes):
                node.children[child_pos] = child

            split_end_t = perf_counter()

            logging.debug(
                "\t\tDP naive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t)
            )

            continue

        elif operation == Operation.CREATE_LEAF:
            if not op_params:
                if f"({local_data})" in dataset_slices["row"]:
                    dataset_slices["row"][f"({local_data})"] +=1
                else:
                    dataset_slices["row"][f"({local_data})"] = 1



            counter["leaf"][scope[0]] += 1
            dataset_slices["leaf"][scope[0]].append(local_data)
            leaf_start_t = perf_counter()
            node = create_leaf(local_data, ds_context, scope, epsilon, bounds, accountant)

            parent.children[children_pos] = node
            leaf_end_t = perf_counter()

            logging.debug(
                "\t\t DP created leaf {} for scope={} (in {:.5f} secs)".format(
                    node.__class__.__name__, scope, leaf_end_t - leaf_start_t
                )
            )

        else:
            raise Exception("Invalid operation: " + operation)

    node = root.children[0]
    assign_ids(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err
    node = Prune(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err


    c = []

    for i in dataset_slices["row"].keys():
        if len(dataset_slices["row"]) != 0:
            c.append(dataset_slices["row"][i])


    #print(c)
    return node, np.max(c)
