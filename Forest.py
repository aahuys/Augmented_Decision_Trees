__author__ = 'Aaron Huys'
#import
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DecisionTrees import *

############## class containing random forest ##############
class Forest:
    #initialization
    #inputs = number of trees in forest, datapartition technique, amount of rated children
    #       minimal depth, maximum depth, tree regeneration stop, tree split function
    #       adaptive depths (decide depth of each tree based on size data partition)
    #       interval split function, amount of datasubsplits in extended tree
    def __init__(self, df, numberTrees, DataPartition, width, minDepth, maxDepth,
                regenStop, determineSplit, intervalSplit, adaptive_depths = False,
                dfSplits=5, seed=None):
        self.df = df
        self.numberTrees = numberTrees
        self.adaptive_depths = adaptive_depths
        self.forest = self.CreateForest(DataPartition,width,minDepth,maxDepth,
                                        regenStop,determineSplit,intervalSplit,
                                        dfSplits,seed)

    def CreateForest(self, DataPartition, width, minDepth, maxDepth, regenStop,
                     determineSplit, intervalSplit, dfSplits,seed):
        #perform datapartition
        dataParts = DataPartition(self.df,self.numberTrees,seed)
        forest = []
        if self.adaptive_depths:
            #find sizes of partitions and normalize
            partSizes = np.array([p.shape[0] for p in dataParts])
            partSizes = partSizes/np.sum(partSizes)
            #find depths for each tree
            totalMaxDepth = maxDepth*self.numberTrees
            totalMinDepth = maxDepth*self.numberTrees
            maxDepths = partSizes*totalMaxDepth
            minDepths = partSizes*totalMinDepth
            for i,part in enumerate(dataParts):
                minD = round(minDepths[i])
                maxD = round(maxDepths[i])
                tree = ExtendedDecisionTree(part,width,minD,maxD,
                                            determineSplit,intervalSplit,regenStop
                                            ,part,dfSplits=dfSplits)
                forest.append(tree)
        else:
            #iterate over different dataset to create tree
            for i,part in enumerate(dataParts):
                tree = ExtendedDecisionTree(part,width,minDepth,maxDepth,
                                            determineSplit,intervalSplit,regenStop
                                            ,part,dfSplits=dfSplits)
                forest.append(tree)
        return forest

############ class containing data partitioning ############
class DataPartitioning:
    #function to randomly split the dataframe based on users
    #inputs = dataframe, number of subsets, seed (default results in 1)
    #output = list with item-split distinct subsets
    def RandomPartition(df, number,seed=None):
        #get items
        items = df['item'].unique()
        #results in same split of dataset for different tries
        if seed == None:
            seed = 1
        else:
            seed = np.random.randint(1,1000)
        rng = np.random.RandomState(seed)
        #shuffle items
        rng.shuffle(items)
        subsets = []
        size = (items.shape[0])//number
        #append subsets to dfs
        for i in range(number):
            if i == number+1:
                subItems = items[i*size:]
            subItems = items[i*size:(i+1)*size]
            temp = df[df['item'].isin(subItems)]
            subsets.append(temp)
        print('partitioning done')
        return subsets
    
    #function to cluster-based split the dataframe
    #inputs = dataframe, number of subsets, seed (default results in 1)
    #output = list with item-split distinct subsets
    def ClusteredPartition(df, partitions,seed=1):
        #create a movie-item matrix
        movieItemMatrix = (df.pivot(index='item', columns='user', values='rating')
                             .fillna(0))
        #using matrif factorization to find collaborative features
        nrFeatures = 30       #hard coded
        nmf = NMF(n_components=nrFeatures)
        movieFeatures = nmf.fit_transform(movieItemMatrix)
        #use ratings as feature as well
        ratingFeature = df.groupby('item')['rating'].mean().reset_index()
        ratingFeature = ratingFeature.rename(columns={'rating': 'avg_rating'})    
        #combinte ratings
        combinedFeatures = np.concatenate((movieFeatures, ratingFeature.iloc[:, 1:]), axis=1)
        #scale features
        scaler = StandardScaler()
        scaledFeatures = scaler.fit_transform(combinedFeatures)
        #apply k-means clustering
        kmeans = KMeans(n_clusters=partitions)
        kmeans.fit(scaledFeatures)
        #add cluster column
        ratingFeature['cluster'] = kmeans.labels_
        #merge cluster label with original dataset
        moviesClustered = df.merge(ratingFeature, on='item')
        #divide dataset
        subsets = []
        for i in range(partitions):
            subsets.append(moviesClustered[moviesClustered['cluster']==i])
        print('partitioning done')
        return subsets
    
################# stand alone functions #################
def readForest(forest,user,df):
    #init items
    items = []
    #read items for every tree
    for tree in forest.forest:
        items += readTree(tree,user,df)
    return items