import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

print("#======= Data Preparation =======#")
df = pd.read_csv("data/cluster_data.csv", header=None)
X = df.values

RESOURCES_DIRECTORY = "resources"
CLUSTER_COUNT = 4

def make_scatterplot(X=X):
    # Plot the input data
    # a. creates figure
    plt.figure()
    plt.title('Input data')

    # b. get the range of X and Y (long way)
    # first column
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()

    # second column
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    # same as above: get the range of X and Y (short way)
    # x_min, x_max = X[:, 0].min(), X[:, 0].max()
    # y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # c. set plot limits
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)

    # plot the points
    plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='black', s=30)
    plt.savefig(f'{RESOURCES_DIRECTORY}/scatterplot_1.jpeg', bbox_inches='tight')
    plt.close()

def make_scatterplot_with_boundaries(X=X):
    kmeans = KMeans(init='k-means++', n_clusters=CLUSTER_COUNT, n_init=20)
    # init: method of experimemtal finding the initial location of the centroids
    # n_init: the algorithm will run n_init times with different cetroids and the best result of those will be taken

    # Train the KMeans clustering model
    kmeans.fit(X)
    # we need a grid of points for drawing a smooth border between clusters
    # define step size of the mesh 
    step_size = 0.01

    # we need to cover all points of our data
    # create a grid out of an array of X values and an array of y values

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()

    # second column
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    x_coord = np.arange(x_min, x_max, step_size)
    y_coord = np.arange(y_min, y_max, step_size)

    # meshgrid() creates a matrix of coordinates from the two vectors of coordinates
    x_vals, y_vals = np.meshgrid(x_coord, y_coord)

    # Predict cluster labels for all the points on the grid 
    # ravel() returns 1D-array
    xx = x_vals.ravel()
    yy = y_vals.ravel()

    # np.c_ concatenates the arguments
    labels = kmeans.predict(np.c_[xx,yy])
    labels = labels.reshape(x_vals.shape)
    print("Labels:\n", labels)


    # Plot the clusters
    # create new plot area
    plt.figure()
    # clear the plot area
    plt.clf()

    plt.title('Boundaries of clusters')

    # plot the frame
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # plot the clusters 
    plt.imshow(labels, interpolation='nearest',
            extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
            cmap='viridis', aspect='auto')
    # plot the points as they belong to the clusters
    plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='white', s=30) 

    # plot the centroids
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1],  s=200, linewidths=2, color='yellow', marker='*', zorder=3, facecolors='black')

    # annotate the centroids
    for i, center in enumerate(centers):
        plt.annotate(i, center+[0.0,1.0], 
                    size=15, zorder=1, color='yellow', weight='bold', 
                    horizontalalignment='center', verticalalignment='center',) 
    plt.savefig('resources/scatterplot_with_boundaries.jpeg', bbox_inches='tight')
    plt.close()



## Determine K by Elbow and Silouette Methods
# Determine k by minimizing the distortion - 
# the sum of the squared distances between each observation vector and its centroid


def make_wcss(one, two, X):
    K = range(one,two)
    distortions = []
    for k in K:
        model = KMeans(n_clusters=k).fit(X)
        model.fit(X)
        distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]) 
    return distortions



def scatter_elbow(X=X):
    print("> Generated Scatterplot")
    #make_scatterplot(X)


    print("\n======= Step 3: Implement KMeans Algorithm for Training a Prediction Model ======")
    wcss = make_wcss(1, 10, X)

    print("> Generated WCSS:\n", wcss)

    plt.plot(range(1, 10), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('resources/scatterplot_elbow.jpeg', bbox_inches='tight')
    plt.close()


def cluster_stuff():
    # Optimal number of clusters K
    CLUSTER_COUNT = 4  # In our case it's 4

    # Create an instance of KMeans classifier
    kmeans = KMeans(init='k-means++', n_clusters=CLUSTER_COUNT, n_init=20)
    # init: method of experimemtal finding the initial location of the centroids
    # n_init: the algorithm will run n_init times with different cetroids and the best result of those will be taken

    # Train the KMeans clustering model
    kmeans.fit(X)
    print("> Kmeans", kmeans)

    print("\n====== Implement the Trained Model for Prediction =======")
    y_pred = kmeans.predict(X)
    print("> y predicted:\n", y_pred)   

    # See the predicted labels of clusters
    # cluster labels are stored in variable 'kmeans.labels_'
    print("> kmeans labels:\n", kmeans.labels_)
    arr = []
    print("\n======= Review the Results =======")
    # Split the clusters, for demo purpose only
    for i in range(CLUSTER_COUNT):
        # slice the cluster
        cluster = X[y_pred == i]    
        # print the shape
        print("Cluster ", i, ": ", cluster.shape)    
        # plot the points of this cluster
        plt.scatter(cluster[:, 0], cluster[:, 1])   
        plt.grid(True)
        plt.title("Cluster " +  str(i) +  ": " +  str(cluster.shape))
        file_name = f'scatterplot_cluster_{str(i)}.jpeg'
        plt.savefig("resources/" + file_name, bbox_inches='tight')
        arr.append(file_name)
        plt.close()
    return arr




# Plot the points with color code of the predicted clusters
# viridis - a color map, https://matplotlib.org/users/colormaps.html
# s - marker size, here, the dots size
# c - marker color, here taken from the predicted clusters
def scatterplot_all_clustered(X=X):# Create an instance of KMeans classifier
    # Optimal number of clusters K
    CLUSTER_COUNT = 4  # In our case it's 4
    kmeans = KMeans(init='k-means++', n_clusters=CLUSTER_COUNT, n_init=20)
    kmeans.fit(X)
    print("> All Clusters in One Plot")
    plt.scatter(X[:,0], X[:,1])
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red')
    plt.title('All Clusters in One Plot w/ centers')
    print(kmeans.cluster_centers_)
    plt.savefig('resources/scatterplot_with_all_clusters.jpeg', bbox_inches='tight')
    plt.close()

#print("\n======= Print Bondaries of Clusters ========")
#make_scatterplot_with_boundaries(X)

#print("\n======== Store the model in a file ========")
#joblib.dump(kmeans, 'model/kmmodel.pkl')