import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, filemode='w', format='\n%(asctime)s\n%(message)s')
logging.Formatter('\n%(asctime)s - %(message)s')

def visualize_clusters(clusters, centroids):

    # fig, ax = plt.subplots()
    # scatter-plot of centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='s', c='#F008', label='Centroids')
    # for i in range(len(centroids)):
    #     plt.annotate(f'c{i}', (centroids[0,0], centroids[0,1]))
    rng = np.random.default_rng()
    # generate random color values for data points in each cluster
    colors = rng.random(size=(n_clusters, 4), dtype=np.float32)
    # opacity set to 0.5 for all points
    colors[:, 3] = 1
    for i in range(n_clusters):
        # scatter-plot of all data points belonging to different clusters
        plt.scatter(clusters[i][:, 0], clusters[i]
                    [:, 1], marker='.', color=tuple(colors[i]))
    # plt.pause(2)
    plt.xlim((0, count))
    plt.ylim((0, count))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Result')
    plt.legend()
class KMeansClustering:
    def runKMeans(self, X: np.ndarray, n_clusters: int, n_iterations: int) -> (list, np.array):
        '''
        The KMeans clustering algorithm.
        Returns:
        clusters: list of np.ndarrays of clusters.
        centroids: np.array of size = n_clusters
        '''
        self.n_clusters = n_clusters
        self.init_centroids(X)
        for i in range(n_iterations):
            self.allocate(X)
            self.update_centroids()
        if X.shape[1] == 2:
            visualize_clusters(self.clusters, self.centroids)
        # logging.debug(f'KMeans results after allocate:\nX:{X}\nclusters:{self.clusters}\nlen:{len(self.clusters)}\nlen1:{len(self.clusters[0])}\nlen2:{len(self.clusters[1])}')
        return self.clusters, self.centroids
    
    def init_centroids(self, X: np.ndarray):
        '''
        Initialize centroids with random examples (or points) from the dataset.
        '''
        #Number of examples
        l = X.shape[0]
        #Initialize centroids array with points from X with random indices chosen from 0 to number of examples
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(l, size=self.n_clusters, replace=False)]
        # self.centroids = X[np.random.randint(0, l, size=self.n_clusters)]
        self.centroids.astype(np.float32)

    
    def allocate(self, X: np.ndarray):
        '''
        This function forms new clusters from the centroids updated in the previous iterations.
        '''
        #Step 1: Fill new clusters with a single point
        #Calculate the differences in the features between X and centroids using broadcast subtract 
        res = X - self.centroids[:, np.newaxis]
        logging.debug(res.shape)    #(n_clusters, X.shape[0], X.shape[1])
        
        #Find Euclidean distances using the above differences
        euc = np.linalg.norm(res, axis = 2)
        # euc = np.array([[1,1,3],[4,2,3],[6,7,8]], dtype=np.float32)
        m, n = euc.shape
        #Add the closest point to the corresponding centroid to the cluster array.
        #We do this to avoid formation of empty clusters
        # res = np.where(euc == euc.min(axis=1)[:, np.newaxis])  #indices of the first entered points
        first_indices = np.full((m,), -1, dtype=int)

        while True:
            res = np.argmin(euc, axis=1)
            res.astype(int)
            resu = np.unique(res)
            l = len(res)
            lu = len(resu)
            if lu == l:
                first_indices = res
                break
            else:
                arr = []
                for i in range(l): 
                    # logging.debug(f'arr entries: {np.where(res==i)}, i:{i}')
                    arr.append(np.where(res==i)[0])    #!DOUBTFUL
                # logging.debug(f'euc:{euc}\nres:{res}\narr:{arr}')
                for i in range(l): 
                    if len(arr[i]) == 1:
                        first_indices[arr[i]] = i
                        # logging.debug(f'fi after equal:\nfi:{first_indices}, i:{i}')
                    elif len(arr[i]) > 1:
                        # logging.debug(f'euc entries for arr[i]={arr[i]} and i={i}\neuc[arr[i], i]:{euc[arr[i], i]}')
                        temp = np.argmin(euc[arr[i], i])    #!DOUBTFUL
                        first_indices[temp] = i
            # logging.debug(f'argmin:{np.argmin(euc, axis=1)}\neuc:{euc}\neuc.shape:{euc.shape}\nres(first_indices): {res}\nres2: {res2}\neuc[res]: {euc[res2]}\n')           
            if -1 in first_indices:
                # euc = euc[np.where(first_indices == -1)]
                col_ind = np.nonzero(np.isin(np.arange(n),first_indices))
                euc[:, col_ind[0]] = euc[col_ind[0], :] = np.inf
                m,n = euc.shape
                # logging.debug(f'euc in while:{euc}\nnp.isin(np.arange(n), first_indices):{ np.nonzero(np.isin(np.arange(n),first_indices))}')
            else:
                break
        # logging.debug(f'euc:{euc}\nfirst_indices on completion:{first_indices}')
        cluster_array = X[first_indices]
        cluster_array = list(np.expand_dims(cluster_array, axis=1))

        #Step 2: Allocate the remaining points to the closest clusters
        #Calculate the differences in the features between centroids and X using broadcast subtract 
        res = self.centroids - X[:, np.newaxis]
        # logging.debug(res.shape)    #(X.shape[0], n_clusters, X.shape[1])

        #Find Euclidean distances of each point with all centroids 
        euc = np.linalg.norm(res, axis=2)

        #Find the closest centroid from each point. 
        # Find unique indices of the closest points. Using res again for optimization
        #not unique indices
        res =  np.where(euc == euc.min(axis=1)[:, np.newaxis])  
        #res[0] is used as indices for row-wise indices in res[1]
        min_indices = res[1][np.unique(res[0])]   
        # logging.debug(f'len(min_indices)={len(min_indices)}')
        #Set first indices to -1 to avoid adding them
        min_indices[first_indices] = -1
        # logging.debug(f'len(min_indices)={len(min_indices)}')
        for i, c in enumerate(min_indices):
            if not c == -1:
                cluster_array[c] = np.append(cluster_array[c], [X[i]], axis=0)    #add the point to the corresponding cluster
        # if len(X) == 2 and (cluster_array[0].shape == (2,2) or cluster_array[1].shape == (2,2)):
        # logging.debug(f'first_indices: {first_indices}\nmin_indices: {min_indices}\ncentroids: {self.centroids}')
        #update the fair clusters array 
        self.clusters = cluster_array
    
    def update_centroids(self):
        '''
        This function updates the centroids based on the updated clusters.
        '''
        #Make a rough copy
        centroids = self.centroids
        #Find mean for every cluster
        for i in range(self.n_clusters):
            centroids[i] = np.mean(self.clusters[i], axis=0)
        
        #Update fair copy 
        self.centroids = centroids

def numConcat(li: list):
    '''
    Utility function to concatenate two float numbers in a list.
    Reference: https://www.geeksforgeeks.org/python-program-to-concatenate-two
    -integer-values-into-one/
    '''
    num1, num2 = li
    # Convert both the numbers to strings
    num1 = str(int(num1))
    num2 = str(int(num2))
    # Concatenate the strings
    num1 += num2
    return int(num1)

# Visualization available
n_clusters = 10
count = 100
X = np.empty((0, 2))
rng = np.random.default_rng()
i = 0
while i < count:
    xy = rng.choice(count, 2, replace=False)
    if numConcat(xy) not in list(map(numConcat, X)):
        X = np.append(X, [xy], axis=0)
    else:
        i -= 1
    i += 1

start = perf_counter()
KMC = KMeansClustering()
clusters, centroids = KMC.runKMeans(X, n_clusters, 7)
stop = perf_counter()
# logging.debug(f'clusters = {clusters}\ncentroids = {centroids}')
logging.debug(f'Elapsed Time: {stop-start}')
plt.show()