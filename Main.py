__author__ = 'Aaron Huys'
#imports
import multiprocessing

from InitData import *
from DecisionTrees import *
from Evaluation import *
from ParallelTrees import *

def simpleTree():
    #set parameters
    total_dataFraction = 10
    numberRatedChildren = 2
    dynamicdepth = 4
    depth = 8
    splitFunc = SplitFunction.logPopEntropySplit
    intervalFunc = IntervalFunction.getIntervals
    train_testFraction = 0.2

    #timing
    startTime = time.time()
    #init data
    df = InitRatingsSubSet(total_dataFraction)
    df = df.rename(columns={'userId': 'user', 'movieId': 'item'})
    print('loading data done')
    #split data
    train_set, test_set = splitActiveLearningData(df, train_testFraction)
    print('data split')
    #if splitFunc is entropy based, entropy needs to be calculated
    #before start of building tree
    if splitFunc == SplitFunction.logPopEntropySplit:
        train_set = entropyCalculation(train_set,)
        #for trees with more than 2 rated children
        if numberRatedChildren != 2:
            train_set = multiEntropyCalculation(train_set,numberRatedChildren)
    #init tree
    # you can create the tree here. Choose from types found in DecisionTrees.py or Forest.py
    tree = DynamicDecisionTree(train_set
                               ,numberRatedChildren
                               ,dynamicdepth
                               ,depth
                               ,splitFunc
                               ,intervalFunc)
    print('tree build')
    print('Tree construction time: '+str(time.time()-startTime))
    #predict use correct prediction algorithm, choose based on consstructed tree:
        #predictTreebatch
        #predictForestbatch
        #predictWebbatch
    preds, zeroRatingUsers, averageSolicitedItems = predictTreebatch(train_set, test_set, tree, 0)
    #get metrics
    RMSE = getRMSE(preds)
    MAE = getMAE(preds)
    AUC = getROC(preds,train_set)
    print('\n')
    print('\n')
    print('Total users with no solicited ratings: '+str(len(zeroRatingUsers)))  
    print('Average amount of solicited ratings: '+str(averageSolicitedItems)) 
    print('RMSE :\t'+str(RMSE))
    print('MAE :\t'+str(MAE))
    print('AUC :\t'+str(AUC))
    print('Total computation time = '+str(time.time()-startTime))
    return

def parallelTree():
    #set parameters
    total_dataFraction = 25
    depth = 17     #largest depth within reasonable time (40+hours) 
    train_testFraction = 0.2

    #timing
    startTime = time.time()
    #init data
    df = InitRatingsSubSet(total_dataFraction)
    df = df.rename(columns={'userId': 'user', 'movieId': 'item'})
    print('loading data done')
    #split data
    train_set, test_set = splitActiveLearningData(df, train_testFraction)
    print('data split')

    #get upper tree structure (15 leaves)
    #pickled version is needed in getExtendedTree
    upperTree = getStartPoints()

    #get parallel trees locally
    #to optimize, every iteration should be performed on a separate machine

    ############################### IMPORTANT ###############################
    #depths are hardcoded in function ParallelTrees.__init__ and numberToPath
    #to change, increment or decrement self.mindepth and self.maxdepth 
    #########################################################################

    subtrees = []
    for id in range(16):
        tree = ParallelTrees(id, train_set)
        subtrees.append(tree)
    #after computing on different machines, trees can be loaded using this script:
    #trees should be store in map pickled_trees under correct names
    #loadSubTrees(depth)

    #combine all trees using:
    tree = getExtendedTree(depth)
    print('tree build')
    print('Tree construction time: '+str(time.time()-startTime))
    #predict use correct prediction algorithm, choose based on consstructed tree:
    preds, zeroRatingUsers, averageSolicitedItems = predictTreebatch(train_set, test_set, tree, 0)
    #get metrics
    RMSE = getRMSE(preds)
    MAE = getMAE(preds)
    AUC = getROC(preds,train_set)
    print('\n')
    print('\n')
    print('Total users with no solicited ratings: '+str(len(zeroRatingUsers)))  
    print('Average amount of solicited ratings: '+str(averageSolicitedItems)) 
    print('RMSE :\t'+str(RMSE))
    print('MAE :\t'+str(MAE))
    print('AUC :\t'+str(AUC))
    print('Total computation time = '+str(time.time()-startTime))
    return



if __name__ == '__main__':
    multiprocessing.freeze_support()
    #simple tree
    simpleTree()

    #parallel tree
    #parallelTree()
