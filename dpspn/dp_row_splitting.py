"""
Code inspired by the SPFlow library (https://github.com/SPFlow/SPFlow)
"""

import numpy as np
from diffprivlib import models as dp


from spn.algorithms.splitting.Base import split_data_by_clusters, preproc
from spn.algorithms.splitting.RDC import rdc_transformer
from spn.structure.StatisticalTypes import MetaType


def get_bounds(ds_context, scope):
    mi = ds_context.domains[scope[0]][0]
    ma = ds_context.domains[scope[0]][-1]
    for entry in range(len(scope)):
        domain = ds_context.domains[entry]
        mi = min(mi,domain[0])
        ma =max(ma, domain[-1])
    return(mi,ma)


def get_split_rows_dp_KMeans(epsilon, bounds, accountant,n_clusters=2, pre_proc=None, ohe=False ):
    def split_rows_KMeans(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        if MetaType.REAL in meta_types:

            data = rdc_transformer(
                local_data,
                meta_types,
                domains,
                k=10,
                s=1.0/6.0,
                non_linearity=np.sin,
                return_matrix=True,
                rand_gen=None,
            )

            bounds=(min([min(data[i]) for i in range(len(data))]),max([max(data[i]) for i in range(len(data))]))
            #bounds=(-0.5,0.5)
            #bounds=(-(1/12)*np.sqrt(2*np.log(len(local_data))), (1/12)*np.sqrt(2*np.log(len(local_data))))
            print(bounds)
            print((1/12)*np.sqrt(2*np.log(len(local_data))))
        else:

            data = preproc(local_data, ds_context, pre_proc, ohe)

            bounds= get_bounds(ds_context, scope)


        clusters = dp.KMeans(n_clusters=n_clusters, epsilon=epsilon, bounds =bounds, accountant=accountant).fit_predict(data)


        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans