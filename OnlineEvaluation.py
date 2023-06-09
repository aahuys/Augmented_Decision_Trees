__author__ = 'Aaron Huys'

#imports:
from DecisionTrees import *
from Evaluation import *
from InitData import *

import pandas as pd
import numpy as np
import pickle

#function to load data of csv file into pandas dataframe
#input = path to data file and path to used binay tree
#output = 3 dataframes: stage1,stage2,test
def getOnlineEvalDatasets(csvpath, treepath):
    with open(csvpath, newline='') as csvfile:
        #hardcode user count to max id of movielens to avoid overlap
        count = 162541
        dfStage1 = []
        dfStage2 = []
        dfTest = []
        for row in csvfile:
            count +=1
            user = count
            #get email
            email = row.split(',')[0]
            #get ratings of first stage
            ratFirstStage = eval(row.split(',"')[1][:-1])
            if len(ratFirstStage) != 15:
                print("error for "+email)
                break
            #get values if user has seen items
            seenFirstStage = eval(row.split('","')[1])
            #get ratings of second stage
            secondStage = row.split('","')[2][:-2]
            if secondStage[-1]!=']':
                secondStage+=']'
            secondStage = eval(secondStage)
            #second active learning model
            ratSecondStage = secondStage[:15]
            #testset
            ratTestStage = secondStage[15:]
            #get movie ids of queried items
            itemsFirstStage = getMoviesBinaryTree(ratFirstStage,treepath)
            #hardcoded ids of second stage
            itemsSecondStage = [318, 356, 296, 593, 2571, 260, 480, 527,
                                2959, 110, 589, 1196, 1, 50, 4993]
            #test item ids
            itemsTestStage = [1704, 10, 2858, 7361, 5349, 349, 592, 3147,
              293, 253, 1682, 1961, 1206, 2028, 457, 1197, 58559, 4226,
              1265, 1270, 1089, 2628, 858, 33794, 1097]
            #create first df
            for i in range(len(itemsFirstStage)):
                dfStage1.append([user,itemsFirstStage[i],ratFirstStage[i],seenFirstStage[i]])
                
            #create second df
            for i in range(len(itemsSecondStage)):
                dfStage2.append([user,itemsSecondStage[i],ratSecondStage[i]])
            #create test df
            for i in range(len(itemsTestStage)):
                if ratTestStage[i]!=None:
                    dfTest.append([user,itemsTestStage[i],ratTestStage[i]])
        #change to pandas dataframe
        df1 = pd.DataFrame(dfStage1, columns=['user','item','rating','seen'])
        df2 = pd.DataFrame(dfStage2, columns=['user','item','rating'])
        dfT = pd.DataFrame(dfTest, columns=['user','item','rating'])
        return df1,df2,dfT

#function to get the items for each users ratings
def getMoviesBinaryTree(ratings, treepath):
    items = []
    #read in used decision tree
    with open(treepath, 'rb') as f:
        tree = pickle.load(f)
    #get root
    currentNode = tree.root
    for r in ratings:
        items.append(currentNode.item)
        if r < 3.5:
            currentNode = currentNode.lchild
        else:
            currentNode = currentNode.rchild
    return items

#function to predict all the seen items of the active learning of stage 1
#input = movielens ratings, obtain ratings active learning set, test set
#output is the predictions of the model using active learning
def getBinaryPredsStage1Seen(perma_set,train_set,test_set):
    #filter out only seen movies
    filtered_train_set = train_set[train_set['seen']==True][['user','item','rating']]
    zeroRatingUsers = set(train_set['user'].unique())-set(filtered_train_set['user'].unique())
    averageSolicitedItems = filtered_train_set.shape[0]/len(train_set['user'].unique())
    #appended filtered with perma train set to create trainset of model
    total_train_set = pd.concat([perma_set, filtered_train_set])
    #filter out users that don't provide rating in active learning
    test_set = test_set[test_set['user'].isin(filtered_train_set['user'].unique())]
    #create and fit recommender system
    knn_algo = knn.UserUser(20,feedback='explicit')
    knn_algo.fit(total_train_set)
    #get predictions
    with contextlib.redirect_stdout(None):
        # Call the code that generates the FutureWarning here
        preds = batch.predict(knn_algo,test_set,verbose=False)
    return preds,zeroRatingUsers,averageSolicitedItems

#function to predict all the items of the active learning of stage 2
#input = movielens ratings, obtain ratings active learning set, test set
#output is the predictions of the model using active learning
def getBinaryPredStage2(perma_set,train_set,test_set):
    #filter out only rated movies
    filtered_train_set = train_set.dropna()
    zeroRatingUsers = set(train_set['user'].unique())-set(filtered_train_set['user'].unique())
    averageSolicitedItems = filtered_train_set.shape[0]/len(train_set['user'].unique())
    #appended filtered with perma train set to create trainset of model
    total_train_set = pd.concat([perma_set, filtered_train_set])
    #filter out users that don't provide rating in active learning
    test_set = test_set[test_set['user'].isin(filtered_train_set['user'].unique())]
    #create and fit recommender system
    knn_algo = knn.UserUser(20,feedback='explicit')
    knn_algo.fit(total_train_set)
    #get predictions
    with contextlib.redirect_stdout(None):
        # Call the code that generates the FutureWarning here
        preds = batch.predict(knn_algo,test_set,verbose=False)
    return preds,zeroRatingUsers,averageSolicitedItems

#function to predict all the items of the active learning of stage 1
#modification of the ratings shoulld be performed before calling this function
#input = movielens ratings, obtain ratings active learning set, test set
#output is the predictions of the model using active learning
def getBinaryPredStage1(perma_set,train_set,test_set):
    #filter out only rated movies
    filtered_train_set = train_set.drop('seen', axis=1)
    zeroRatingUsers = set(train_set['user'].unique())-set(filtered_train_set['user'].unique())
    averageSolicitedItems = filtered_train_set.shape[0]/len(train_set['user'].unique())
    #appended filtered with perma train set to create trainset of model
    total_train_set = pd.concat([perma_set, filtered_train_set])
    #filter out users that don't provide rating in active learning
    test_set = test_set[test_set['user'].isin(filtered_train_set['user'].unique())]
    #create and fit recommender system
    knn_algo = knn.UserUser(20,feedback='explicit')
    knn_algo.fit(total_train_set)
    #get predictions
    with contextlib.redirect_stdout(None):
        # Call the code that generates the FutureWarning here
        preds = batch.predict(knn_algo,test_set,verbose=False)
    return preds,zeroRatingUsers,averageSolicitedItems

#transform rating more to average
def transformRating(rating,strength):
    #center around 0
    translatedR = rating-3.5
    #normalize
    normedR = translatedR/5
    #create transition factor based on strength
    factor = (1-strength)**(-abs(normedR))-1
    #check for left or right translation
    if rating > 3.5:
        return rating - factor
    else:
        return rating + factor

#transform ratings for unseen movies in dataset∑
def transformDataset(stage1,strength):
    df = stage1.copy()
    df['rating'] = df.apply(lambda x: transformRating(x['rating'],strength) 
                            if not x['seen'] 
                            else x['rating'], axis=1)
    return df