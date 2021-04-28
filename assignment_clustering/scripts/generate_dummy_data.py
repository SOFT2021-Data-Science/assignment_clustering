import numpy as np
from sklearn.datasets import make_blobs


# Generate random dummy cluster data and save it as a csv file under "data"
if __name__ == "__main__":
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    np.savetxt("data/cluster_data.csv", X, delimiter=",")