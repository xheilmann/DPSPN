import numpy as np
import csv
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType


def extract_metatypes(features):
    metatypes = []
    for entry in range(len(features)):
        type = features[entry][1]

        if type =="binary":
            metatypes.append(MetaType.BINARY)
        elif type == "categorical" or type == "discrete":
            metatypes.append(MetaType.DISCRETE)
        else:
            metatypes.append(MetaType.REAL)
    return metatypes

def extract_domain(features, metatypes):
    domains = []
    for entry in range(len(features)):
        meta_type = metatypes[entry]
        domain = features[entry][2].split(sep = ",")
        m = domain[-1].rstrip(".")
        d=np.array([float(domain[0]), float(m)])
        if meta_type == MetaType.REAL or meta_type == MetaType.BINARY:
            domains.append(d)
        elif meta_type == MetaType.DISCRETE:
            domains.append(np.arange(d[0], d[1] + 1, 1))
        else:
            raise Exception("Unkown MetaType " + str(meta_type))


    return domains

def get_context(feature_data_path):
    features = list(csv.reader(open(feature_data_path, 'r'), delimiter=':'))
    meta_types = extract_metatypes(features)
    ds_context = Context(meta_types=meta_types)
    ds_context.domains = extract_domain(features, meta_types)
    return ds_context