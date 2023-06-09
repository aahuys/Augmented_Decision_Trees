__author__ = 'Aaron Huys'
#imports
import pandas as pd
import numpy as np
import pickle

from InitData import *
from DecisionTrees import *
from Evaluation import *

#function to  create an upper level decision tree
#tree contains 19 nodes places over 2 depths
#those nodes are the startpoints for the creation of
#the parallelized trees
#tree gets pickled
#input = None
#Non
def getStartPoints():
    #load dataset
    ratingsData = InitRatingsSubSet(25)
    #rename colums
    ratingsData = ratingsData.rename(columns={'userId': 'user', 'movieId': 'item'})
    #split data
    train_set, _ = splitActiveLearningData(ratingsData,0.2)
    #calculate initial entropies
    train_set = entropyCalculation(train_set,10)
    #create upper level
    upperTree = DynamicDecisionTree(train_set,
                                     2,
                                     1,
                                     3,
                                     SplitFunction.logPopEntropySplit,
                                     IntervalFunction.getIntervals,
                                     )
    #visualize tree
    visualizeTree(upperTree,title=False)
    #dump tree in pickle
    with open('upperTree.pkl', 'wb') as f:
        pickle.dump(upperTree, f)
    return upperTree


#simple item holder
#allows compatibility with trees
class ItemHolder:
    def __init__(self,items):
        self.items = items
        
#class for subtree that will be parallel
#computed along other
class ParallelTrees:
    def __init__(self, id,df):
        self.df = df
        self.id = id
        self.mindepth = 9
        self.maxdepth = 15
        self.nodedepth = 1
        self.itemHolder = None
        self.tree = self.createParallelTree(id,df)  #create tree
        self.pickle = self.pickleWrite()            #pickle tree
    
    #create the tree
    #input = id of subtree and total train_set
    def createParallelTree(self,id,df):
        new_df = self.getStartPointDataset(id,df)
        #create tree
        #add df,items and depths of subtree
        tree = ExtendedDecisionTree(self.df,
                                    2,
                                    self.mindepth,
                                    self.maxdepth,
                                    SplitFunction.logPopEntropySplit,
                                    IntervalFunction.getIntervals,
                                    self.maxdepth-2,
                                    new_df,
                                    self.itemHolder,
                                    )
        return tree
    #creates pickle of final subtree
    def pickleWrite(self):
        title = 'subTree_'+str(self.id)+'.pkl'
        with open(title, 'wb') as f:
            pickle.dump(self.tree, f)
    
    #get node for path
    def getNode(self,root,path):
        node = root
        for edge in path[:-1]:
            node = node.children[edge]
        return node
    
    #get items in node
    def getItems(self,node):
        temp = node
        items = []
        #iterate to top
        while temp!=None:
            items.append(temp.item)
            temp = temp.parent
        #reverse to go from root to leaf
        items.reverse()
        return items
    
    #get dataframe of nodes after modification have occured during traversal
    #input =  original dataframe, items traversed, path (like, dislike,unkown) list
    #ouput = modified dataframe and item to propose next
    def getDataFrame(self,df,items,path):
        temp = df
        #iterate over path to recreate dataframe for start points
        for i,item in enumerate(items[:-1]):
            edge = path[i]
            interval = (IntervalFunction.getIntervals(2)+[[None]])[edge]
            if edge !=2:
                #find rated users
                users = temp[temp['item'] == item].reset_index(drop=True)
                users = users[users['rating'].isin(interval)]['user']
            else:
                #find unrated users
                all_users = set(temp['user'].unique())
                rated_users = set(temp[temp['item'] == item]['user'].unique())
                users = list(all_users - rated_users)
            #create dataset for node, all users above, and without item
            child_dataset = temp[temp['item']!=item]
            temp = child_dataset[child_dataset['user'].isin(users)]
        #repeat for last item
        item = SplitFunction.logPopEntropySplit(temp)
        edge = path[-1]
        interval = (IntervalFunction.getIntervals(2)+[[None]])[edge]
        if edge !=2:
            #find rated users
            users = temp[temp['item'] == item].reset_index(drop=True)
            users = users[users['rating'].isin(interval)]['user']
        else:
            #find unrated users
            all_users = set(temp['user'].unique())
            rated_users = set(temp[temp['item'] == item]['user'].unique())
            users = list(all_users - rated_users)
        #create dataset for node, all users above, and without item
        child_dataset = temp[temp['item']!=item]
        temp = child_dataset[child_dataset['user'].isin(users)]
        return temp, item

    #get start dataframe for parallel tree
    #input = leaf number, original dataframe
    #output = modified dataset
    def getStartPointDataset(self, point,df):
        with open('upperTree.pkl', 'rb') as f:
            tree = pickle.load(f)
        path = self.numberToPath(point)
        node = self.getNode(tree.root, path)
        items = self.getItems(node)
        #place all items in holder
        self.itemHolder = ItemHolder(items)
        tempdf, item = self.getDataFrame(df,items,path)
        items.append(item)
        return tempdf
    
    #get path for number
    #hardcoded
    def numberToPath(self,i):
        if i==1:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,15
            return [0,0]
        elif i==2:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,15 
            return [0,1]
        elif i==3:
            self.nodedepth=1
            self.mindepth, self.maxdepth = 8,15 
            return [0,2]
        elif i==4:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,15 
            return [1,0]
        elif i==5:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,15 
            return [1,1]
        elif i==6:
            self.nodedepth=1
            self.mindepth, self.maxdepth = 8,15 
            return [1,2]
        elif i==7:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,14
            return [2,0,0]
        elif i==8:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,14
            return [2,0,1]
        elif i==9:
            self.nodedepth=1
            self.mindepth, self.maxdepth = 8,14 
            return [2,0,2]
        elif i==10:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,14 
            return [2,1,0]
        elif i==11:
            self.nodedepth=2
            self.mindepth, self.maxdepth = 7,14 
            return [2,1,1]
        elif i==12:
            self.nodedepth=1
            self.mindepth, self.maxdepth = 8,14 
            return [2,1,2]
        elif i==13:
            self.nodedepth=1
            self.mindepth, self.maxdepth = 8,14 
            return [2,2,0]
        elif i==14:
            self.nodedepth=1
            self.mindepth, self.maxdepth = 8,14 
            return [2,2,1]
        else:
            self.nodedepth=0
            self.mindepth, self.maxdepth = 9,14 
            return [2,2,2]

#load pickle files of all subtrees for a extended tree of certain depth
def loadSubTrees(depth=17):
    subtrees = []
    for i in range(1,16):
        path = 'pickled_trees/parallel_trees_d'+str(depth)+'/subTree_'+str(i)+'.pkl'
        with open(path, 'rb') as f:
        # Load the pickled data from the file
            tree = pickle.load(f)
            tree.df = None
            tree.dfs = None
            subtrees.append(tree)
    return subtrees

#function that combine all pickle files and returns a extended width decision tree
#this is to evaluate the tree
def getExtendedTree(depth):
    base_tree = None
    with open('pickled_trees/upperTree.pkl','rb') as f:
        base_tree = pickle.load(f)
    subtrees = loadSubTrees(depth)
    #hardcoded integration of subtrees
    base_tree.root.children[0].children[0] = subtrees[0].root #1
    subtrees[0].root.chosenInterval = [0.5,1,1.5,2,2.5,3]
    subtrees[0].root.parent = base_tree.root.children[0]

    base_tree.root.children[0].children[1] = subtrees[1].root #2
    subtrees[1].root.chosenInterval = [3.5,4,4.5,5]
    subtrees[1].root.parent = base_tree.root.children[0]

    base_tree.root.children[0].children[2] = subtrees[2].root #3
    subtrees[2].root.chosenInterval = [None]
    subtrees[2].root.parent = base_tree.root.children[0]

    base_tree.root.children[1].children[0] = subtrees[3].root #4
    subtrees[3].root.chosenInterval = [0.5,1,1.5,2,2.5,3]
    subtrees[3].root.parent = base_tree.root.children[1]

    base_tree.root.children[1].children[1] = subtrees[4].root #5
    subtrees[4].root.chosenInterval = [3.5,4,4.5,5]
    subtrees[4].root.parent = base_tree.root.children[1]

    base_tree.root.children[1].children[2] = subtrees[5].root #6
    subtrees[5].root.chosenInterval = [None]
    subtrees[5].root.parent = base_tree.root.children[1]

    base_tree.root.children[2].children[0].children[0] = subtrees[6].root #7
    subtrees[6].root.chosenInterval = [0.5,1,1.5,2,2.5,3]
    subtrees[6].root.parent = base_tree.root.children[2].children[0]

    base_tree.root.children[2].children[0].children[1] = subtrees[7].root #8
    subtrees[7].root.chosenInterval = [3.5,4,4.5,5]
    subtrees[7].root.parent = base_tree.root.children[2].children[0]

    base_tree.root.children[2].children[0].children[2] = subtrees[8].root #9
    subtrees[8].root.chosenInterval = [None]
    subtrees[8].root.parent = base_tree.root.children[2].children[0]

    base_tree.root.children[2].children[1].children[0] = subtrees[9].root #10
    subtrees[9].root.chosenInterval = [0.5,1,1.5,2,2.5,3]
    subtrees[9].root.parent = base_tree.root.children[2].children[1]

    base_tree.root.children[2].children[1].children[1] = subtrees[10].root #11
    subtrees[10].root.chosenInterval = [3.5,4,4.5,5]
    subtrees[10].root.parent = base_tree.root.children[2].children[1]

    base_tree.root.children[2].children[1].children[2] = subtrees[11].root#12
    subtrees[11].root.chosenInterval = [None]
    subtrees[11].root.parent = base_tree.root.children[2].children[1]

    base_tree.root.children[2].children[2].children[0] = subtrees[12].root #13
    subtrees[12].root.chosenInterval = [0.5,1,1.5,2,2.5,3]
    subtrees[12].root.parent = base_tree.root.children[2].children[2]

    base_tree.root.children[2].children[2].children[1] = subtrees[13].root #14
    subtrees[13].root.chosenInterval = [3.5,4,4.5,5]
    subtrees[13].root.parent = base_tree.root.children[2].children[2]

    base_tree.root.children[2].children[2].children[2] = subtrees[14].root #15
    subtrees[14].root.chosenInterval = [None]
    subtrees[14].root.parent = base_tree.root.children[2].children[2]
    return base_tree