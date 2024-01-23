import argparse

import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn

from spn.algorithms.Statistics import get_structure_stats_dict

from dpspn.dpspn import learn_dp_mspn

import pandas as pd
from dpspn.features import get_context

import diffprivlib as dp


#log=logging.getLogger()
#log.setLevel(logging.DEBUG)
acc4 = dp.BudgetAccountant()




parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path', required=True)
parser.add_argument('--test-data-path', required=True)
parser.add_argument('--feature-data-path', required=True, help='Path to feature-metatypes and bounds')
parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon differential privacy parameter')
parser.add_argument('--threshold', type=float, default=0.1, help='Column splitting variable for non-dp SPN')
parser.add_argument('--spn-slice', type=int, default=200, help='Minimal number of instances in data splits in SPN')
parser.add_argument('--dpspn-slice', type=int, default=200, help='Minimal number of instances in data splits in DPSPN')
parser.add_argument('--cols', default=None, help='Column splitting method for DPSPN')
parser.add_argument('--total-operations', type=int, default=10, help='Maximal number of operations on each data slice')



opt = parser.parse_args()

spn_slice = opt.spn_slice
dpspn_slice = opt.dpspn_slice
threshold = opt.threshold
e = opt.epsilon
tot_op = opt.total_operations
cols= opt.cols
train = pd.read_csv(opt.train_data_path)
test = pd.read_csv(opt.test_data_path)
data_columns = [col for col in train.columns]


train = train.values.astype("float")
test = test.values.astype("float")

ds_context = get_context(opt.feature_data_path)
results = []


#learn the non-dp-private SPN
mspn = learn_mspn(train, ds_context, cols="rdc", threshold=threshold, min_instances_slice=spn_slice)
mspn_ll=np.mean(log_likelihood(mspn,test))







dp_spn, max_op_on_dataset_slice = learn_dp_mspn(train, ds_context,  min_instances_slice=dpspn_slice, epsilon=e,  total_operations=tot_op - 1)

layers = get_structure_stats_dict(dp_spn)["layers"]
max_epsilon = e * tot_op
true_epsilon = acc4.total()[0]


#evaluate DPSPN

dpspn_ll=np.mean(log_likelihood(dp_spn,test))

print(f"MSPN_ll:{mspn_ll} \n DPSPN_ll:{dpspn_ll}")
print(get_structure_stats_dict(mspn), get_structure_stats_dict(dp_spn))














