__author__ = 'Aaron Huys'
#imports
import numpy as np
import pandas as pd
import contextlib
import lenskit.algorithms.user_knn as knn
from lenskit import batch
from lenskit.metrics.predict import rmse,mae
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import time

from DecisionTrees import readTree,readWebOfTrees,readWebOfTreesModified
from Forest import readForest

#function to split the dataframe based on user
#80% of users will always be in train set and 20% users will participate in active learning
#input = pandas data frame of ratings.csv of movielens, split between test and train
#output = 2 dataframes, a train set and a test set
# df should have columns 'user', 'rating'
def splitActiveLearningData(df, split=0.2):
    #get a randomly shuffled array of all users in dataset
    all_users = df['user'].unique()
    rng = np.random.RandomState(1)
    rng.shuffle(all_users)
    #create set of user to train and users to test
    train_users = all_users[:int(len(all_users)*(1-split))]
    test_users = all_users[int(len(all_users)*(1-split)):]
    #get rating dataset for those user groups
    train_set = df[df['user'].isin(train_users)]
    test_set = df[df['user'].isin(test_users)]
    return train_set, test_set

#function to calculate predictions of a test set, using a decision tree for active learning
#the function adds the questioned items to a permanent train set, fits the model to this set
#input = train set df, test set df, decision tree from DecisionTrees.py,
#the amount of users to be predicted every batch 
#if batchsize = 0, no batch mode and all predictions at once
#output is the predictions of the model using active learning
# df should have columns 'user', 'rating'
def predictTreebatch(perma_train_set, test_set, tree, batch_size=0):
    startTime = time.time()
    #get a batch of x users out of the test users, predict for this group based on interviewed items
    all_preds = []
    test_users = test_set['user'].unique()
    #init measure vars
    zeroRatingUsers, solicitedItems = [], 0
    count = 0
    #no batches
    if batch_size==0:
        batch_size = len(test_users)
    for i in range(0, len(test_users), batch_size):
        #verbose of function
        count += 1
        print('\tBatch nr '+str(count)+' started')
        knn_algo = None
        #reset train_set to original (permanent) train_set
        train_set = perma_train_set
        #empty lists to append all batch predictions and queried ratings in
        pred_data = []
        tot_queried_data = []
        #users in current batch
        user_batch = test_users[i:i+batch_size]
        for user in user_batch:
            #get interviewed items
            queried_items = readTree(tree,user,test_set)
            #ratings for asked items of user
            data_user = test_set[test_set['user']==user]
            queried_data = data_user[data_user['item'].isin(queried_items)]
            #store amount of solicited items
            solicitedItems += queried_data.shape[0]
            #check if user has rated items:
            if queried_data.shape[0] == 0:
                #if no rated items, add user to zeroRatingUsers
                zeroRatingUsers.append(user)
                continue
            #ratings for not queried items of user, needs to be predicted
            to_predict_data = data_user[~data_user['item'].isin(queried_items)]
            #append pred datas and queried data
            pred_data.append(to_predict_data)
            tot_queried_data.append(queried_data)
        #concat all items that need to be predicted all queried data
        pred_data = pd.concat(pred_data)
        train_set = pd.concat([train_set]+tot_queried_data)
        #create model    
        knn_algo = knn.UserUser(20,feedback='explicit')
        knn_algo.fit(train_set)
        #predict ratings other items of user batch
        with contextlib.redirect_stdout(None):
            # Call the code that generates the FutureWarning here
            preds = batch.predict(knn_algo,pred_data,verbose=False)
        all_preds.append(preds)
    #return concatenation of all predictions of batches
    averageSolicitedItems = solicitedItems/len(test_users)
    print('\tPrediction computation time: '+str(time.time()-startTime))
    return pd.concat(all_preds), zeroRatingUsers, averageSolicitedItems

#function that returns the RMSE for all non-NaN values
#input = prediction dataframe (output of predictALbatch)
#output = float that is RMSE
def getRMSE(preds):
    return rmse(preds['prediction'], preds['rating'])

#function that returns the modified RMSE
#input = prediction dataframe (output of predictALbatch), test_set dataframe,
#list with users that did not rate item during elicitation
#output = float that is RMSE
def getModifiedRMSE(preds,test_set,zeroRatingUsers):
    #get amount of non-nan preds
    totalPreds = preds.shape[0]
    performedPreds = preds.dropna()
    #all items for users that did not rate elicited item
    nonQueriedUserData = test_set[test_set['user'].isin(zeroRatingUsers)].shape[0]
    #number of false preds = number of nans of predictions 
    # + all items for users that did not rate elicited item
    numberNans = (totalPreds-performedPreds.shape[0]) + nonQueriedUserData
    #get rmse for non-nan preds
    oldRMSE = getRMSE(performedPreds)
    #get modified RMSE
    newRMSE = (1.5833*numberNans+oldRMSE*performedPreds.shape[0])/(totalPreds+nonQueriedUserData)
    return newRMSE

#function that returns the MAE for all non-NaN values
#input = prediction dataframe (output of predictALbatch)
#output = float that is MAE
def getMAE(preds):
    return mae(preds['prediction'],preds['rating'])

#function that returns the modified MAE
#input = prediction dataframe (output of predictALbatch), test_set dataframe,
#list with users that did not rate item during elicitation
#output = float that is MAE
def getModifiedMAE(preds,test_set,zeroRatingUsers):
    #get amount of non-nan preds
    totalPreds = preds.shape[0]
    performedPreds = preds.dropna()
    #all items for users that did not rate elicited item
    nonQueriedUserData = test_set[test_set['user'].isin(zeroRatingUsers)].shape[0]
    #number of false preds = number of nans of predictions 
    # + all items for users that did not rate elicited item
    numberNans = (totalPreds-performedPreds.shape[0]) + nonQueriedUserData
    #get mae for non-nan preds
    oldMAE = getMAE(performedPreds)
    #get modified MAE
    newMAE = (1.5833*numberNans+oldMAE*performedPreds.shape[0])/(totalPreds+nonQueriedUserData)
    return newMAE

#function that plots the roc curve and returns the auc
#input = pandasdataframe of predictions, train_data and path to store graph
#output = the float:auc
def getROC(preds,train_data,path=0):
    df = preds.copy()
    #only roc possible if prediction is made
    df = df.dropna()
    #get the mean of the train data
    meanRating = np.mean(train_data['rating'].to_numpy())
    #create a column that denotes it the user likes the item
    #like is rating > mean, dislike is rating < mean
    df['like'] = df['rating'].apply(lambda x: x>=meanRating)
    df['pred_norm'] = df['prediction'].apply(lambda x: x/5)
    #get ROC curves
    labels = df['like'].to_numpy()
    predictions = df['pred_norm'].to_numpy()
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    #get AUC
    aucScore = round(auc(fpr, tpr),3)
    #plot
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = '+str(aucScore)+')')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if path!=0:
        plt.savefig(path)
    plt.show()
    #return AUC
    return aucScore

#function to calculate predictions of a test set, using a forest for active learning
#the function adds the questioned items to a permanent train set,
# fits the model to this set
#input = train set df, test set df, decision tree from DecisionTrees.py,
#the amount of users to be predicted every batch 
#if batchsize = 0, no batch mode and all predictions at once
#output is the predictions of the model using active learning
# df should have columns 'user', 'rating'
def predictForestbatch(perma_train_set, test_set, forest, batch_size=0):
    startTime = time.time()
    #get a batch of x users out of the test users, predict 
    #for this group based on interviewed items
    all_preds = []
    test_users = test_set['user'].unique()
    #init measure vars
    zeroRatingUsers, solicitedItems = [], 0
    count = 0
    #no batches
    if batch_size==0:
        batch_size = len(test_users)
    for i in range(0, len(test_users), batch_size):
        #verbose of function
        count += 1
        print('\tBatch nr '+str(count)+' started')
        knn_algo = None
        #reset train_set to original (permanent) train_set
        train_set = perma_train_set
        #empty lists to append all batch predictions and queried ratings in
        pred_data = []
        tot_queried_data = []
        #users in current batch
        user_batch = test_users[i:i+batch_size]
        for user in user_batch:
            #get interviewed items
            queried_items = readForest(forest,user,test_set)
            #ratings for asked items of user
            data_user = test_set[test_set['user']==user]
            queried_data = data_user[data_user['item'].isin(queried_items)]
            #store amount of solicited items
            solicitedItems += queried_data.shape[0]
            #check if user has rated items:
            if queried_data.shape[0] == 0:
                #if no rated items, add user to zeroRatingUsers
                zeroRatingUsers.append(user)
                continue
            #ratings for not queried items of user, needs to be predicted
            to_predict_data = data_user[~data_user['item'].isin(queried_items)]
            #append pred datas and queried data
            pred_data.append(to_predict_data)
            tot_queried_data.append(queried_data)
        #concat all items that need to be predicted all queried data
        pred_data = pd.concat(pred_data)
        train_set = pd.concat([train_set]+tot_queried_data)
        #create model    
        knn_algo = knn.UserUser(20,feedback='explicit')
        knn_algo.fit(train_set)
        #predict ratings other items of user batch
        with contextlib.redirect_stdout(None):
            # Call the code that generates the FutureWarning here
            preds = batch.predict(knn_algo,pred_data,verbose=False)
        all_preds.append(preds)
    #return concatenation of all predictions of batches
    averageSolicitedItems = solicitedItems/len(test_users)
    print('\tPrediction computation time: '+str(time.time()-startTime))
    return pd.concat(all_preds), zeroRatingUsers, averageSolicitedItems

#function to calculate predictions of a test set, using a web of trees for active learning
#the function adds the questioned items to a permanent train set, fits the model to this set
#input = train set df, test set df, decision tree from DecisionTrees.py,
#the amount of users to be predicted every batch 
#if batchsize = 0, no batch mode and all predictions at once
#output is the predictions of the model using active learning
# df should have columns 'user', 'rating'
def predictWebbatch(perma_train_set, test_set, WOT, batch_size=0):
    startTime = time.time()
    #get a batch of x users out of the test users, 
    #predict for this group based on interviewed items
    all_preds = []
    test_users = test_set['user'].unique()
    #init measure vars
    zeroRatingUsers, solicitedItems = [], 0
    count = 0
    #no batches
    if batch_size==0:
        batch_size = len(test_users)
    for i in range(0, len(test_users), batch_size):
        #verbose of function
        count += 1
        print('\tBatch nr '+str(count)+' started')
        knn_algo = None
        #reset train_set to original (permanent) train_set
        train_set = perma_train_set
        #empty lists to append all batch predictions and queried ratings in
        pred_data = []
        tot_queried_data = []
        #users in current batch
        user_batch = test_users[i:i+batch_size]
        for user in user_batch:
            #get interviewed items
            queried_items = readWebOfTrees(WOT,user,test_set)
            #ratings for asked items of user
            data_user = test_set[test_set['user']==user]
            queried_data = data_user[data_user['item'].isin(queried_items)]
            #store amount of solicited items
            solicitedItems += queried_data.shape[0]
            #check if user has rated items:
            if queried_data.shape[0] == 0:
                #if no rated items, add user to zeroRatingUsers
                zeroRatingUsers.append(user)
                continue
            #ratings for not queried items of user, needs to be predicted
            to_predict_data = data_user[~data_user['item'].isin(queried_items)]
            #append pred datas and queried data
            pred_data.append(to_predict_data)
            tot_queried_data.append(queried_data)
        #concat all items that need to be predicted all queried data
        pred_data = pd.concat(pred_data)
        train_set = pd.concat([train_set]+tot_queried_data)
        #create model    
        knn_algo = knn.UserUser(20,feedback='explicit')
        knn_algo.fit(train_set)
        #predict ratings other items of user batch
        with contextlib.redirect_stdout(None):
            # Call the code that generates the FutureWarning here
            preds = batch.predict(knn_algo,pred_data,verbose=False)
        all_preds.append(preds)
    #return concatenation of all predictions of batches
    averageSolicitedItems = solicitedItems/len(test_users)
    print('\tPrediction computation time: '+str(time.time()-startTime))
    return pd.concat(all_preds), zeroRatingUsers, averageSolicitedItems

#modified version to allow easy evaluation by passing extra arguments
def predictWebbatchModified(perma_train_set, test_set, WOT, max_ratings, max_queries, batch_size=0):
    startTime = time.time()
    #get a batch of x users out of the test users, predict for this group based on interviewed items
    all_preds = []
    test_users = test_set['user'].unique()
    #init measure vars
    zeroRatingUsers, solicitedItems = [], 0
    count = 0
    #no batches
    if batch_size==0:
        batch_size = len(test_users)
    for i in range(0, len(test_users), batch_size):
        #verbose of function
        count += 1
        print('\tBatch nr '+str(count)+' started')
        knn_algo = None
        #reset train_set to original (permanent) train_set
        train_set = perma_train_set
        #empty lists to append all batch predictions and queried ratings in
        pred_data = []
        tot_queried_data = []
        #users in current batch
        user_batch = test_users[i:i+batch_size]
        for user in user_batch:
            #get interviewed items
            queried_items = readWebOfTreesModified(WOT,user,test_set,max_ratings,max_queries)
            #ratings for asked items of user
            data_user = test_set[test_set['user']==user]
            queried_data = data_user[data_user['item'].isin(queried_items)]
            #store amount of solicited items
            solicitedItems += queried_data.shape[0]
            #check if user has rated items:
            if queried_data.shape[0] == 0:
                #if no rated items, add user to zeroRatingUsers
                zeroRatingUsers.append(user)
                continue
            #ratings for not queried items of user, needs to be predicted
            to_predict_data = data_user[~data_user['item'].isin(queried_items)]
            #append pred datas and queried data
            pred_data.append(to_predict_data)
            tot_queried_data.append(queried_data)
        #concat all items that need to be predicted all queried data
        pred_data = pd.concat(pred_data)
        train_set = pd.concat([train_set]+tot_queried_data)
        #create model    
        knn_algo = knn.UserUser(20,feedback='explicit')
        knn_algo.fit(train_set)
        #predict ratings other items of user batch
        with contextlib.redirect_stdout(None):
            # Call the code that generates the FutureWarning here
            preds = batch.predict(knn_algo,pred_data,verbose=False)
        all_preds.append(preds)
    #return concatenation of all predictions of batches
    averageSolicitedItems = solicitedItems/len(test_users)
    print('\tPrediction computation time: '+str(time.time()-startTime))
    return pd.concat(all_preds), zeroRatingUsers, averageSolicitedItems