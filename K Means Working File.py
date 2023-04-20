import numpy as np
import sklearn
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

digits = load_digits()
data = scale(digits.data) #features being scaled down to reduce euclidean distance
y = digits.target

k = 10 # could make it dynamic by using len(np.unique(y)) instead of using an integer here
samples, features = data.shape

#mathematics behind the metrics can be found on the sklearn clusters documentation
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

#classifier parameters
clf = KMeans(n_clusters=k, init="random", n_init=20, max_iter=500)
bench_k_means(clf, "1", data)
