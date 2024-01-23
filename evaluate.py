import argparse
import os
import time
import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.MPE import mpe, mpe_prod, mpe_sum, get_mpe_top_down_leaf
from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.algorithms.LearningWrappers import learn_classifier
from dpspn.dpspn import learn_dp_mspn
from dpspn.dpspn import learn_dp_classifier
import pandas as pd
from dpspn.features import get_context
from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
import diffprivlib as dp

#log=logging.getLogger()
#log.setLevel(logging.DEBUG)
acc4 = dp.BudgetAccountant()


def print_save_results(results, path):

    spn_data = pd.DataFrame(results, columns=['name', "max_epsilon", "true_epsilon",'MSPN-AUC', 'DPSPN-AUC', 'MSPN-AUPRC', 'DPSPN-AUPRC'])
    print(spn_data)
    if not os.path.isfile(opt.result_data_path):
        raise Exception('Result path does not exist')
    spn_data.to_csv(path, index=False, mode="a")

    print("Saved results at : ", path)



parser = argparse.ArgumentParser()
parser.add_argument('--target-variable',required=True, help='Required if data has a target class')
parser.add_argument('--train-data-path', required=True)
parser.add_argument('--test-data-path', required=True)
parser.add_argument('--feature-data-path', required=True, help='Path to feature-metatypes and bounds')
parser.add_argument('--result-data-path', required=True, help='.csv to store results')
parser.add_argument("--iterations", type=int, default=1, help="Number of times the dp algorithm is performed")
parser.add_argument('--enable-privacy', action='store_true', help='Enable private data generation')
parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon differential privacy parameter')
parser.add_argument('--delta', type=float, default=0.0, help='Delta differential privacy parameter')
parser.add_argument('--threshold', type=float, default=0.1, help='Column splitting variable for non-dp SPN')
parser.add_argument('--spn-slice', type=int, default=200, help='Minimal number of instances in data splits in SPN')
parser.add_argument('--dpspn-slice', type=int, default=200, help='Minimal number of instances in data splits in DPSPN')
parser.add_argument('--cols', default=None, help='Column splitting method for DPSPN')
parser.add_argument('--total-operations', type=int, default=10, help='Maximal number of operations on each data slice')
parser.add_argument('--save-synthetic', action='store_true', help='Save the synthetic data into csv')
parser.add_argument('--output-data-path', help='Required if synthetic data needs to be saved')


opt = parser.parse_args()

spn_slice = opt.spn_slice
dpspn_slice = opt.dpspn_slice
threshold = opt.threshold
e = opt.epsilon
delta = opt.delta
tot_op = opt.total_operations
cols= opt.cols
train = pd.read_csv(opt.train_data_path)
test = pd.read_csv(opt.test_data_path)
data_columns = [col for col in train.columns]

test_labels = test[opt.target_variable].values.copy()
test[opt.target_variable] = [np.nan for _ in range((test.shape[0]))]
target_index = train.columns.get_loc(opt.target_variable)

train = train.values.astype("float")
test = test.values.astype("float")

ds_context = get_context(opt.feature_data_path)
results = []
def histogram_top_down(node, input_vals, lls_per_node, data=None):
    get_mpe_top_down_leaf(node, input_vals, data=data, mode=histogram_mode_top(node))


def histogram_mode_top(node):
    areas = np.diff(node.breaks) * node.densities
    _x = np.argmax(areas)
    mode_value = node.bin_repr_points[_x]
    if node.scope[0]==target_index:
        mode_value = node.densities[1]
    return mode_value

#learn the non-dp-private SPN
mspn = learn_classifier(train, ds_context, learn_mspn, target_index, rows="rdc", cols="binary_random", threshold=threshold, min_instances_slice=spn_slice)
pred_mpe = mpe(mspn, test, node_top_down_mpe={Product: mpe_prod, Sum: mpe_sum, Histogram: histogram_top_down})
pred = pred_mpe[:,target_index]
m_auroc = roc_auc_score(test_labels, pred)
m_auprc = average_precision_score(test_labels, pred)



if not opt.enable_privacy:
    if delta == 0.0:
        dp_mspn, max_op_on_dataset_slice = learn_dp_classifier(train, ds_context, learn_dp_mspn, target_index, min_instances_slice=dpspn_slice, epsilon=e, total_operations=tot_op - 1, cols=cols )

        layers = get_structure_stats_dict(dp_mspn)["layers"]
        max_epsilon = e * tot_op
        true_epsilon = acc4.total()[0]

    else:
        dp_mspn, max_op_on_dataset_slice = learn_dp_classifier(train, ds_context, learn_dp_mspn, target_index, min_instances_slice=dpspn_slice, epsilon=e,  total_operations=tot_op - 1)

        layers = get_structure_stats_dict(dp_mspn)["layers"]
        max_epsilon = e * tot_op
        true_epsilon = acc4.total()[0]

else:
    # generate synthetic dp data
    sample = [np.nan for _ in range(len(train[0]))]
    train_class = np.delete(train, target_index, axis=1)
    test_class = np.delete(test, target_index, axis=1)
    best_perf = 0.0
    dp_auc_iter=[]
    DP_start=time.time()
    for it in range(opt.iterations):
        dp_mspn, max_op_on_dataset_slice = learn_dp_classifier(train, ds_context, learn_dp_mspn, target_index, min_instances_slice=dpspn_slice, epsilon=e, total_operations=tot_op - 1, cols=cols)
        samples = np.array(sample * train.shape[0]).reshape(-1, len(sample))
        dp_samples = sample_instances(dp_mspn, samples, RandomState(123))
        syn_y = dp_samples[:, target_index]
        syn_x = np.delete(dp_samples, target_index, axis=1)
        unique, counts = np.unique(syn_y, return_counts=True)

        if len(counts) > 1:
            learner = LogisticRegression(random_state=RandomState(123))
            score = learner.fit(syn_x,syn_y)
            pred_probs = learner.predict_proba(train_class)[:, 1]
            temp_perf = roc_auc_score(train[:,target_index], pred_probs)
            dp_auc_iter.append(temp_perf)
        else:
            dp_auc_iter.append(temp_perf)
            continue


        # Select best synthetic data
        if temp_perf > best_perf:
            best_perf = temp_perf.copy()
            synth_train_data = dp_samples.copy()

        print('Iteration: ' + str(it + 1))
        print('DPSPN-Best-Perf:' + str(best_perf))
    DP_end=time.time()
    DP_time = DP_end -DP_start


    max_epsilon = e * tot_op
    true_epsilon = acc4.total()[0]

    #train mspn on dp synthetic data
    dp_syn_mspn_class = learn_classifier(synth_train_data, ds_context, learn_mspn,target_index, cols="rdc", threshold=threshold, min_instances_slice=spn_slice)
    pred_mpe = mpe(dp_syn_mspn_class, test, node_top_down_mpe={Product: mpe_prod, Sum: mpe_sum, Histogram: histogram_top_down})
    pred = pred_mpe[:, target_index]
    sf_auroc = roc_auc_score(test_labels, pred)
    sf_auprc = average_precision_score(test_labels, pred)



    # generate synthetic non-dp data
    samples1 = np.array(sample * train.shape[0]).reshape(-1, len(sample))
    non_dp_samples = sample_instances(mspn, samples1, RandomState(123))
    #train mspn on non-dp synthetic data
    syn_mspn_class = learn_classifier(non_dp_samples, ds_context, learn_mspn, target_index, cols="rdc", threshold=threshold, min_instances_slice=spn_slice)
    pred_mpe = mpe(syn_mspn_class, test, node_top_down_mpe={Product: mpe_prod, Sum: mpe_sum, Histogram: histogram_top_down})
    pred = pred_mpe[:, target_index]
    s_auroc = roc_auc_score(test_labels, pred)
    s_auprc = average_precision_score(test_labels, pred)

    results.append(["synthetic-SPN",max_epsilon, true_epsilon, s_auroc, sf_auroc,  s_auprc, sf_auprc])



    if opt.save_synthetic:

        if not os.path.isdir(opt.output_data_path):
            raise Exception('Output directory does not exist')
        true_epsilon= np.round(true_epsilon,2)

        X_syn_df = pd.DataFrame(data=synth_train_data, columns=data_columns)
        X_syn_df.to_csv(opt.output_data_path + f"/synthetic_data_{true_epsilon}.csv", index=False,
                        float_format='%.1f')
        print("Saved synthetic data at : ", opt.output_data_path)

#evaluate DPSPN
pred_mpe = mpe(dp_mspn, test, node_top_down_mpe={Product: mpe_prod, Sum: mpe_sum, Histogram: histogram_top_down})
pred = pred_mpe[:,target_index]
d_auroc = roc_auc_score(test_labels, pred)
d_auprc = average_precision_score(test_labels, pred)

results.append(["SPN", max_epsilon, true_epsilon, m_auroc,  d_auroc, m_auprc, d_auprc])
print_save_results(results, opt.result_data_path )
print(get_structure_stats_dict(dp_mspn))














