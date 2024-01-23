
from collections import namedtuple
from sys import maxsize

from diffprivlib import tools as dp
import numpy as np
import matplotlib.pyplot as plt
from diffprivlib.mechanisms import GeometricTruncated

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import MetaType, Type
import logging

logger = logging.getLogger(__name__)

from spn.structure.leaves.histogram.Histograms import Histogram






def create_dp_histogram_leaf(data, ds_context, scope,  epsilon, range,  accountant, alpha=0.5, hist_source="numpy"):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    data = data[~np.isnan(data)]

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]
    domain = ds_context.domains[idx]

    assert not np.isclose(np.max(domain), np.min(domain)), "invalid domain, min and max are the same"
    range = (np.min(domain), np.max(domain))


    if data.shape[0] == 0 or (np.var(data) == 0 and meta_type == MetaType.REAL):
        #one datapoint or all nones
        maxx = np.max(domain)
        minx = np.min(domain)
        breaks = np.array([minx, maxx])
        densities = np.array([1])
        repr_points = np.array([(maxx) / 2])
        if meta_type == MetaType.DISCRETE or meta_type == MetaType.BINARY:
            repr_points = repr_points.astype(int)


    else:

        breaks, densities, repr_points = getHistogramVals(data, meta_type, domain, epsilon, range, accountant, source=hist_source)

    # laplace smoothing
    if alpha:
        n_samples = data.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert len(densities) == len(breaks) - 1
    #if scope[0] == 1:
        #plt.plot(breaks[:-1], densities, "o", color="green")
        #plt.show()

    densities =densities/np.sum(densities)




    return Histogram(breaks.tolist(), densities.tolist(), repr_points.tolist(), scope=idx, meta_type=meta_type)


def getHistogramVals(data, meta_type, domain,  epsilon, range,  accountant, source="numpy"):


    if meta_type == MetaType.DISCRETE or meta_type == MetaType.BINARY:

        # for discrete, we just have to count
        breaks = np.array([d for d in domain] + [domain[-1] + 1])
        densities, breaks = dp.histogram(data, bins=breaks, density=True, epsilon = epsilon, range = range,  accountant=accountant)
        repr_points = np.asarray(domain)
        return breaks, densities, repr_points


    if source == "numpy":
        #breaks = np.histogram_bin_edges(data.astype("int32"), bins="auto")
        densities, breaks = dp.histogram(data, bins="auto", density=True, epsilon = epsilon, range = range,  accountant=accountant)
        mids = ((breaks + np.roll(breaks, -1)) / 2.0)[:-1]
        return breaks, densities, mids



    assert False, "unkown histogram method " + source
