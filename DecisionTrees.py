__author__ = 'Aaron Huys'
#imports
import pandas as pd
import graphviz
import numpy as np
from InitData import InitMoviesData25M
from concurrent.futures import ThreadPoolExecutor
################# class containing node #################
class Node:
    #initialization
    def __init__(self,item,parent, depth, users, interval):
        self.item = item                     #which item determines the split
        self.parent = parent                 #parent node
        self.children = []                   #list of all children
        self.depth = depth                   #current depth
        self.users = users                   #amount of users in split
        self.chosenInterval = interval       #rating interval chosen for edge
        self.leaf = False
        
    #checks if node is a leaf
    def isLeaf(self):
        return self.leaf
    #set leaf status
    def setLeaf(self,val):
        self.leaf=val
    #add a child, child must be of type class Node
    def addChild(self,child):
        self.children.append(child)
        return None
    
############ class containing  extended node ############
class ExtendedNode:
    #initialization
    def __init__(self,item,parent, depth, users, interval,items):
        self.item = item                     #which item determines the split
        self.parent = parent                 #parent node
        self.children = []                   #list of all children
        self.depth = depth                   #current depth
        self.users = users                   #amount of users in split
        self.chosenInterval = interval       #rating interval chosen for edge
        self.items = items                   #visited items
    #add a child, child must be of type class Node
    def addChild(self,child):
        self.children.append(child)
        return None
    
########### class containing binary tree node ###########
class BinaryNode:
    #initializatoin
    def __init__(self, item, parent, path):
        self.item = item                    #which item determines the split
        self.parent = parent                #parent of node
        self.path = path                    #dislike or like edge (0 or 1)
        self.lchild = None                  #left child
        self.rchild = None                  #right child
    
################# class containing decision tree #################
#based on ID3
class DecisionTree:
    #initalization
    def __init__(self, df, width, stop, determineSplit, intervalSplit):
        self.df = df                           #dataframe of tree
        self.width = width                     #amount of children with rating
        self.stop = stop                       #depth of tree
        self.determineSplit = determineSplit   #splitting function of tree
        self.intervalSplit = intervalSplit     #function to determine interval of ratings in split
        self.root = self.createTree(df, None, 0, None)
    
    #create a decision tree of dataframe df with depth of node as parameter and chosen interval as parameters
    ###inputs###
    #df should be ratingsDataV2
    #parent should be None to initiate
    #depth should initiate at 0, increases each new depth
    #rating interval chosen to reach node, init at None
    #output = decision tree
    def createTree(self, df, parent, depth, interval):
        #print('depth:'+(' '*depth)+str(depth))                      #see process
        #determine item to split on dataset df
        item = self.determineSplit(df)
        all_users = list(df['user'].unique())
        #initiate node with split item and given function parameeters
        root = Node(item, parent, depth, len(all_users), interval)
        #check if item exists
        if item==False:
            root.setLeaf(True)
            return None
        #check if root is leaf
        if depth==self.stop:
            root.setLeaf(True)
            return None
        #create children
        #keep rated users in list
        rated_users = []
        for interval in self.intervalSplit(self.width):
            #get all users that gave item a rating withing interval
            users = df[df['item'] == item].reset_index(drop=True)
            users = users[users['rating'].isin(interval)]['user'].to_list()
            rated_users+=users
            #create dataset for node, consisting of all ratings of users above, and without item
            child_dataset = df[df['item']!=item]
            child_dataset = child_dataset[child_dataset['user'].isin(users)]
            #add child to current node
            root.addChild(self.createTree(child_dataset, root, depth+1, interval))
        #add child with users that did not rate item
        #fast methos to get unrated users
        for i in rated_users:
            all_users.remove(i)
        unrated_users = all_users
        unrated_dataset = df[df['user'].isin(unrated_users)]
        root.addChild(self.createTree(unrated_dataset, root, depth+1, [None]))
        #return the root to be able to travers
        return root
    
################# class containing decision tree with dynamic height #################   
class DynamicDecisionTree:
    #initalization
    def __init__(self, df, width, max_ratings, max_depth, determineSplit, intervalSplit):
        self.df = df                           #dataframe of tree
        self.width = width                     #amount of children with rating
        self.max_ratings = max_ratings+1       #max amount of rated items
        self.max_depth = max_depth             #max depth of path
        self.determineSplit = determineSplit   #splitting function of tree
        self.intervalSplit = intervalSplit     #function to determine interval of ratings in split
        self.root = self.createTree(df, None, 0, 0, None)
        
    #create a decision tree of dataframe df with depth of node as 
    #parameter and chosen interval as parameters
    ###inputs###
    #df should be dataset
    #parent should be None to initiate
    #2 <= width <=5 determines amount of children
    #node_depth should initiate at 0, increases each new depth
    #ratings_depth should initiate at 0, increases each rating (not unknown)
    #rating interval chosen to reach node, init at None
    def createTree(self, df, parent, ratings_depth, node_depth, interval):
        #determine item to split on dataset df
        item = self.determineSplit(df)
        #get all users in df
        all_users = list(df['user'].unique())
        #initiate node with split item and given function parameeters
        root = Node(item, parent, node_depth, len(all_users), interval)
        #check if item exists
        if item==False:
            root.setLeaf(True)
            return None
        #check if root is at max node depth
        if node_depth==self.max_depth:
            root.setLeaf(True)
            return None
        #check if root has max amount of decisions
        if ratings_depth==self.max_ratings:
            root.setLeaf(True)
            return None
        #filter out if last choice is uknown:
        if interval == [None] and ratings_depth+1==self.max_ratings:
            root.setLeaf(True)
            return None
        #create children
        #keep rated users in list
        rated_users = []
        for interval in self.intervalSplit(self.width):
            #get all users that gave item a rating withing interval
            users = df[df['item'] == item].reset_index(drop=True)
            users = users[users['rating'].isin(interval)]['user'].to_list()
            rated_users+=users
            #create dataset for node, consisting of all ratings of users above, and without item
            child_dataset = df[df['item']!=item]
            child_dataset = child_dataset[child_dataset['user'].isin(users)]
            #add child to current node
            root.addChild(self.createTree(child_dataset, root, ratings_depth+1, node_depth+1, interval))
        #add child with users that did not rate item
        #fast methos to get unrated users
        for i in rated_users:
            all_users.remove(i)
        unrated_users = all_users
        unrated_dataset = df[df['user'].isin(unrated_users)]
        root.addChild(self.createTree(unrated_dataset, root, ratings_depth, node_depth+1, [None]))
        #return the root to be able to travers
        return root

################# class containing decision tree with extended width #################   
class ExtendedDecisionTree:
    #initalization
    def __init__(self, df, width, max_ratings, max_depth, determineSplit,
                intervalSplit, regenStop,
                newdf, parent = None, dfSplits=25):
        
        self.df = df                       #dataframe of tree
        self.rng = np.random.RandomState(1)    #set seed
        self.dfs = self.splitDataFrame(df, dfSplits)  #datapartitions
        self.splits = dfSplits                 #number of data partitions
        self.width = width                     #amount of children with rating
        self.max_ratings = max_ratings+1       #max amount of rated items
        self.max_depth = max_depth             #max depth of path
        self.determineSplit = determineSplit   #splitting function of tree
        self.intervalSplit = intervalSplit     #function to determine interval of ratings in split
        self.Ncounter = 0                      #check for number of nodes
        self.regenStop = regenStop             #from wich height stops regeneration
        self.root = self.createTree(newdf, parent, 0, 0, None)  #root of tree
        
    #create a decision tree of dataframe df with depth of node as parameter and chosen interval as parameters
    ###inputs###
    #df should be dataset
    #parent should be None to initiate
    #2 <= width <=5 determines amount of children
    #node_depth should initiate at 0, increases each new depth
    #ratings_depth should initiate at 0, increases each rating (not unknown)
    #rating interval chosen to reach node, init at None
    #prev_items holds all items of the path from the root to the root of the subtree
    def createTree(self, functionDF, parent, ratings_depth, node_depth, interval):
        #see proces
        self.Ncounter +=1
        if self.Ncounter%10000==0:
            print(self.Ncounter//10000)
    #stop conditions
        #check if root is at max node depth
        if node_depth==self.max_depth:
            return None
        #check if root has max amount of decisions
        if ratings_depth==self.max_ratings:
            return None
        #filter out if last choice is uknown:
        if interval == [None] and ratings_depth+1==self.max_ratings:
            return None
        #check previous items
        prev_items = []
        if parent != None:
            prev_items += parent.items
        #check if path can be created
        #if not, generate subtree with dataset of tree if not too low in tree
        if functionDF.shape[0] == 0:
            if node_depth<(self.regenStop):
                #get random index
                idx = self.rng.randint(0,self.splits)
                tempDF = self.dfs[idx]
                functionDF=tempDF[~tempDF['item'].isin(prev_items)]
            else:
                return None
        #determine item to split on dataset df
        item = self.determineSplit(functionDF)
        #add item to list of items of parent
        prev_items += [item]
        #get all users in df
        all_users = set(functionDF['user'].unique())
        #initiate node with split item and given function parameeters
        root = ExtendedNode(item, parent, node_depth, len(all_users), interval, prev_items)
        #create children
        #keep rated users in list
        rated_users = []
        for interval in self.intervalSplit(self.width):
            #get all users that gave item a rating withing interval
            users = functionDF[functionDF['item'] == item].reset_index(drop=True)
            users = users[users['rating'].isin(interval)]['user'].to_list()
            rated_users+=users
            #create dataset for node, consisting of all ratings of users above, and without item
            child_dataset = functionDF[functionDF['item']!=item]
            child_dataset = child_dataset[child_dataset['user'].isin(users)]
            #add child to current node
            root.addChild(self.createTree(child_dataset, root, ratings_depth+1, node_depth+1, interval))
        #add child with users that did not rate item
        #fast methos to get unrated users
        unrated_users = list(all_users - set(rated_users))
        unrated_dataset = functionDF[functionDF['user'].isin(unrated_users)]
        root.addChild(self.createTree(unrated_dataset, root, ratings_depth, node_depth+1, [None]))
        #return the root to be able to travers
        return root
    #functino to split dataset into amount smaller parts
    def splitDataFrame(self, df, amount=25):
        items = df['item'].unique()
        #results in same split of dataset for different tries
        rng = np.random.RandomState(1)
        #shuffle items
        rng.shuffle(items)
        dfs = []
        size = (items.shape[0])//amount
        #append subsets to dfs
        for i in range(amount):
            if i == amount+1:
                subItems = items[i*size:]
            subItems = items[i*size:(i+1)*size]
            temp = df[df['item'].isin(subItems)]
            dfs.append(temp)
        return dfs
########### class containing binary decision  tree with extended width ###########   
class BinaryDecisionTree:
    #initalization
    def __init__(self, df, max_depth, determineSplit,intervalSplit, dfSplits=10):
        self.df = df                                    #dataframe of tree
        self.rng = np.random.RandomState(1)             #seed
        self.dfs = self.splitDataFrame(df, dfSplits)    #subsets of dataframe
        self.splits = dfSplits                          #number of datasubsplits
        self.max_depth = max_depth                      #max number of queries
        self.determineSplit = determineSplit            #splitting function of tree
        self.intervalSplit = intervalSplit              #function to determine interval of ratings in split
        self.Ncounter = 0
        self.root = self.createTree(df, None, 0, None)  #root of tree
    ###inputs###
    #df should be dataset
    #parent should be None to initiate
    #depth should initiate at 0, increases each new depth
    #rating interval chosen to reach node, init at None
    #output = decision tree
    def createTree(self, functionDF, parent, depth, path):
        #see proces
        self.Ncounter +=1
        if self.Ncounter%1000==0:
            print(self.Ncounter//1000)
        #stop condition
        if depth==self.max_depth:
            return None
        #check if dataset is not none, otherwise add items
        if functionDF.shape[0]==0:
            #collect previous items
            prev_items = []
            nodeIt = parent
            while nodeIt != None:
                prev_items.append(nodeIt.item)
                nodeIt = nodeIt.parent
            #gen new dataset
            idx = self.rng.randint(0,self.splits)
            tempDF = self.dfs[idx]
            functionDF=tempDF[~tempDF['item'].isin(prev_items)]
        #determine split item
        item = self.determineSplit(functionDF)  
        #create new node
        node = BinaryNode(item,parent,path)
    #create first child
        #find users that disliked item
        interval = (self.intervalSplit(2))[0]
        users = functionDF[functionDF['item'] == item].reset_index(drop=True)
        users = users[users['rating'].isin(interval)]['user'].to_list()
        #create dataset for node, consisting of all ratings of users above, and without item
        child_dataset = functionDF[functionDF['item']!=item]
        child_dataset = child_dataset[child_dataset['user'].isin(users)]
        #add child to current node      #path is 0 for dislike
        node.lchild = self.createTree(child_dataset,node,depth+1,0)
    #create second child
        #find users that disliked item
        interval = (self.intervalSplit(2))[1]
        users = functionDF[functionDF['item'] == item].reset_index(drop=True)
        users = users[users['rating'].isin(interval)]['user'].to_list()
        #create dataset for node, consisting of all ratings of users above, and without item
        child_dataset = functionDF[functionDF['item']!=item]
        child_dataset = child_dataset[child_dataset['user'].isin(users)]
        #add child to current node      #path is 0 for dislike
        node.rchild = self.createTree(child_dataset,node,depth+1,1)
        return node
    
    #split total frame into subframes
    #input = dataframe, number of subframes
    #output = list with subframes
    def splitDataFrame(self, df, amount=10):
        items = df['item'].unique()
        #results in same split of dataset for different tries
        rng = np.random.RandomState(1)
        #shuffle items
        rng.shuffle(items)
        dfs = []
        size = (items.shape[0])//amount
        #append subsets to dfs
        for i in range(amount):
            if i == amount+1:
                subItems = items[i*size:]
            subItems = items[i*size:(i+1)*size]
            temp = df[df['item'].isin(subItems)]
            dfs.append(temp)
        return dfs
    
    #visualize tree
    #input = title (if true titles will be shown in figure)
    #ouput = pdf file with visualization of tree
    def visualize(self,title=True):
        dot = graphviz.Digraph(comment='Decision Tree')
        visited = set()
        #check if titles need to be displayed, if so import data
        movieData = None
        if title:
            movieData = InitMoviesData25M()
        #intern functoin to visit al nodes
        def visit(node):
            #node is none
            if node is None:
                return
            #node already visited
            if node in visited:
                return
            #determine edge label:
            if node.path == 0:
                edge = 'Dislike'
            else:
                edge = 'Like'
            visited.add(node)
            #create node label
            nodelabel = str(node.item)
            if title:
                movieTitle = movieData[movieData['movieId']==node.item]['title'].to_list()[0]
                nodelabel = movieTitle
            dot.node(str(id(node)), label=nodelabel)
            #create edge
            if node.parent is not None:
                dot.edge(str(id(node.parent)), str(id(node)),label=edge)
            #recurrent approach
            visit(node.lchild)
            visit(node.rchild)
        #call recurrent function
        visit(self.root)
        dot.render('decision_tree', view=True)

################ class containing web of trees ################  
class WebOfTrees:
    def __init__(self, df, width, max_ratings, max_queries, 
                min_depth, max_depth,
                determineSplit, intervalSplit, dfSplits=15):
        self.df = df                            #dataset
        self.width = width                      #number of not unknown children
        self.max_ratings = max_ratings          #max number of likes and dislikes
        self.max_queries = max_queries          #max depth
        self.determineSplit = determineSplit    #function to find split item
        self.intervalSplit = intervalSplit      #function to find rating intervals
        self.dfSplits = dfSplits                #number of subtree in web
        self.base = DynamicDecisionTree(df, width, max_ratings, max_queries   #get root
                                        , determineSplit, intervalSplit)
        self.subTrees = self.createSubtrees(min_depth,max_depth,dfSplits)   #get subtrees

    #create list with subtrees
    #input = min depth of subtrees, max_depth and amount
    #outp = list with subTrees
    def createSubtrees(self, min_depth,max_depth,dfSplits):
        #split dataset
        dataframes = self.splitDataFrame(self.df,dfSplits)
        subTrees = []
        #create tree for each dataset
        for df in dataframes:
            tree = DynamicDecisionTree(df,self.width, min_depth,max_depth,
                                       self.determineSplit,self.intervalSplit)
            subTrees.append(tree)
        return subTrees
    
    #split dataframe in parts based on user
    #input = dataframe, number of subframes
    #output = list with subframes
    def splitDataFrame(self, df, amount=15):
        items = df['item'].unique()
        #results in same split of dataset for different tries
        rng = np.random.RandomState(1)
        #shuffle items
        rng.shuffle(items)
        dfs = []
        size = (items.shape[0])//amount
        #append subsets to dfs
        for i in range(amount):
            if i == amount+1:
                subItems = items[i*size:]
            subItems = items[i*size:(i+1)*size]
            temp = df[df['item'].isin(subItems)]
            dfs.append(temp)
        return dfs


############# class containing functions that determine item of node #############
#return a random item of the dataset
class SplitFunction:
    #return a random item of dataset
    #input = pandas dataset
    #output = proposed items itemID
    def randomSplit(df):
        items = list(df['item'].unique())
        if len(items) == 0:
            return False
        index = np.random.randint(len(items))
        return items[index]

    #return the most popular item of the dataset
    #input = pandas dataset
    #output = proposed items itemID
    def popularSplit(df):
        if(df.shape[0]==0):
            return False
        return df['item'].value_counts().idxmax()
    
    #return the  popular random item of the dataset
    #input = pandas dataset
    #output = proposed items itemID
    def popularRandomSplit(df):
        if(df.shape[0]==0):
            return False
        temp = df.groupby('item',as_index=False).size().sort_values('size')
        quant = temp['size'].quantile(0.9)
        temp = temp[temp['size']>=quant]
        index = np.random.randint(temp.shape[0])
        return temp.iloc[index]['item']
    
    #returns the highest log(popular)*entropy item of dataset
    def logPopEntropySplit(df):
        if(df.shape[0]==0):
            return False
        #get counts of every item
        counts = df['item'].value_counts().reset_index()
        counts.rename(columns={'item': 'count', 'index': 'item'}, inplace=True)
        #merge with entropy dataset
        merged = pd.merge(counts,df,on='item')
        #create log(pop)*entropy
        merged['popentr'] = np.log(merged['count'])*merged['entr']
        #find max item
        item = merged.loc[merged['popentr'].idxmax()]['item']
        return int(item)
    
################# class containing functions for the intervals of the children #################
class IntervalFunction:
    #function returns hard coded intervals used to determine dataset of children
    #input = int
    #output = list of list containing intervals
    def getIntervals(width):
        if width == 2:
            #list returns [dislike,like] intervals
            return [[0.5,1,1.5,2,2.5,3],[3.5,4,4.5,5]]
        if width == 3:
            #list returns [dislike, average, like] intervals
            return [[0.5,1,1.5,2,2.5,3],[3.5,4],[4.5,5]]
        if width == 4:
            #list returns [strong dislike, weak dislike, weak like, strong like] intervals
            return [[0.5,1,1.5,2,2.5],[3,3.5],[4],[4.5,5]]
        if width == 5:
            #list returns [strong dislike, weak dislike, average, weak like, strong like] intervals
            return [[0.5,1,1.5,2,2.5],[3],[3.5],[4],[4.5,5]]
        else:
            print('Wrong amount given')
            return None

    #function returns the names of each interval, is used in plotting trees
    #input = int
    #output = hardcoded list with edge names
    def edgeNames(width):
        if width == 2:
            return ['Dislike','Like']
        if width == 3:
            return ['Dislike','Average','Like']
        if width == 4:
            return ['Strong Dislike', 'Weak Dislike', 'Weak Like', 'Strong Like']
        if width == 5:
            return ['Strong Dislike', 'Weak Dislike','Average', 'Weak Like', 'Strong Like']
        else:
            print('Wrong amount given')
            return None
    
    
################# stand alone functions #################
#function to visualize tree, should only be tree with small depth to be helpfull
#input = root of tree, bool title displayed or not
#output = opens file containing visualization of tree
def visualizeTree(tree, title=True):
    dot = graphviz.Digraph(comment='Decision Tree')
    visited = set()
    #check if titles need to be displayed, if so import data
    movieData = None
    if title:
        movieData = InitMoviesData25M()
        
    def visit(node, intervals, names):
        #node is none
        if node is None:
            return
        #node already visited
        if node in visited:
            return
        #determine edge label:
        edge=None
        if node.chosenInterval in intervals:
            edge= names[intervals.index(node.chosenInterval)]
        else:
            edge='Unknown'
        visited.add(node)
        #get amount of users in node
        users = node.users
        #create node label
        nodelabel = str(node.item)+'\nUsers: '+str(users)
        if title:
            movieTitle = movieData[movieData['movieId']==node.item]['title'].to_list()[0]
            nodelabel = movieTitle+'\nUsers: '+str(users)
        dot.node(str(id(node)), label=nodelabel)
        #create edge
        if node.parent is not None:
            dot.edge(str(id(node.parent)), str(id(node)),label=edge)
        #recurrent approach
        for child in node.children:
            visit(child, intervals, names)
    #determine edge names
    names = IntervalFunction.edgeNames(tree.width)
    #call recurrent function
    intervals = tree.intervalSplit(tree.width)
    visit(tree.root, intervals, names)
    dot.render('decision_tree', view=True)

#function to read out the splitting items in the tree
#input = tree, userID and dataframe
#output = list containing the items of each node
def readTree(tree, user, df):
    items = []
    #initiate root
    currentNode = tree.root
    nextNode = currentNode
    #walk through decision tree
    while currentNode!= None:
        #get item of current node and append to list
        item = currentNode.item
        items.append(item)
        #find rating of given user for item of node
        rating = list(df.loc[(df['item']==item) & (df['user']==user), 'rating'])
        if len(rating)==0:      #unknown
            rating = None
        else:                   #rated
            rating = rating[0]
        #check which child needs to be selected based on fact that rating is in interval
        for child in currentNode.children:
            if child==None:
                continue
            if rating in child.chosenInterval:
                nextNode = child
        #if path stops because child is None
        if nextNode == currentNode:
            currentNode = None
        else:
            currentNode = nextNode
    #return items
    return items

#function to calculate the entropys of all the items
#input = train_set pandas dataframe
#output = modified train_set with entropys column
def entropyCalculation(train_set,minpop=10,minentr=0.2):
    #create deep copy
    entr_set = train_set.copy()
    #check if item is liked or disliked
    entr_set['like'] = entr_set['rating'].apply(lambda x: int(x>3))
    #get probability for like or dislike per item
    entr_set = (entr_set
                    .groupby('item',group_keys=True)['like']
                    .value_counts(normalize=True))
    #calculate entropy for like column
    entr_set = (entr_set
                    .groupby('item',group_keys=True)
                    .apply(lambda x: -np.sum(x*np.log2(x)))
                    .reset_index())
    entr_set.rename(columns={'like':'entr'},inplace=True)
    entr_set = entr_set[entr_set['entr']>minentr]
    #get counts of item
    count_set = train_set.copy()
    #get counts of item
    count_set = train_set.copy()
    count_set = count_set['item'].value_counts().reset_index()
    count_set.rename(columns={'item':'count','index':'item'},inplace=True)
    #filter out low popularit items since they will not be selected
    count_set = count_set[count_set['count']>minpop]
    #merge popular items (counts) and entr sets
    temp_set = pd.merge(count_set,entr_set,on="item",how="inner")
    #merge to get ratings, drop count and entr
    temp_set = temp_set.drop(['count'], axis=1)
    temp_set = pd.merge(train_set,temp_set,on='item',how="inner")
    return temp_set

#function to calculate the entropys of all the items based on width
#input = train_set pandas dataframe
#output = modified train_set with entropys column
def multiEntropyCalculation(train_set,width,minpop=10,minentr=0.2):
    #create deep copy
    entr_set = train_set.copy()
    #local function to determine in which rating group rating lies
    #is harcoded based on rating distribution (skewed ratings)
    def checkWidthInterval(rating,w):
        #hardcoded based on the intervals
        if w==2:
            if rating>3:
                return 1
            else:
                return 0
        elif w==3:
            if rating<3.5:
                return 0
            elif rating<4.5:
                return 1
            else:
                return 2
        elif w==4:
            if rating<3:
                return 0
            elif rating<4:
                return 1
            elif rating<4.5:
                return 2
            else:
                return 3
        else:
            if rating<3:
                return 0
            elif rating<3.5:
                return 1
            elif rating<4:
                return 2
            elif rating<4.5:
                return 3
            else:
                return 4
    entr_set['interval'] = entr_set['rating'].apply(lambda x: checkWidthInterval(x,width))
    #get probability for like or dislike per item
    entr_set = (entr_set
                    .groupby('item',group_keys=True)['interval']
                    .value_counts(normalize=True))
    #calculate entropy for like column
    entr_set = (entr_set
                    .groupby('item',group_keys=True)
                    .apply(lambda x: -np.sum(x*np.log2(x)))
                    .reset_index())
    entr_set.rename(columns={'interval':'entr'},inplace=True)
    entr_set = entr_set[entr_set['entr']>minentr]
    #get counts of item
    count_set = train_set.copy()
    #get counts of item
    count_set = train_set.copy()
    count_set = count_set['item'].value_counts().reset_index()
    count_set.rename(columns={'item':'count','index':'item'},inplace=True)
    #filter out low popularit items since they will not be selected
    count_set = count_set[count_set['count']>minpop]
    #merge popular items (counts) and entr sets
    temp_set = pd.merge(count_set,entr_set,on="item",how="inner")
    #merge to get ratings, drop count and entr
    temp_set = temp_set.drop(['count'], axis=1)
    temp_set = pd.merge(train_set,temp_set,on='item',how="inner")
    return temp_set

#find the items of the nodes of a traversal through the web for a user
#input = web of trees, user and test_set
#output = list with queried items
def readWebOfTrees(WOT,user,df):
    items = []
    ratings = []
    visited_trees = []
    #initiate root
    currentNode = WOT.base.root
    nextNode = currentNode
    aqRatings = 0
    #walk through web_of_trees
    while (len(items)!= WOT.max_queries) and (aqRatings != WOT.max_ratings):
        #get item of current node and append to list
        item = currentNode.item
    #check if item has already been queried
        #find rating of already queried item
        if len(ratings)!=0 and item in items:
            rating = ratings[items.index(item)]
        #check if user has rated item
        else:
            #append new queried item
            items.append(item)
            #find rating of given user for item of node
            rating = list(df.loc[(df['item']==item) & (df['user']==user), 'rating'])
            if len(rating)==0:      #unknown
                rating = None
            else:                   #rated
                rating = rating[0]
                aqRatings+=1
            ratings.append(rating)
        #check which child needs to be selected based on fact that rating is in interval
        for child in currentNode.children:
            if child==None:
                continue
            if rating in child.chosenInterval:
                nextNode = child
        #if path stops because child is None
        if nextNode == currentNode:
            #take random tree out of web
            partitionNr = np.random.randint(0,WOT.dfSplits)
            #warning if to much trees have been visited
            if len(visited_trees) > WOT.dfSplits*0.9:
                print('Too much trees visited')
                break
            #find partition of tree that has not been visited yet
            while partitionNr in visited_trees:
                partitionNr = np.random.randint(0,WOT.dfSplits)
            visited_trees.append(partitionNr)
            currentNode = (WOT.subTrees[partitionNr]).root
        else:
            currentNode = nextNode
    #return items
    return items    

#modified version to decide the max ratings and max queries (allows faster evaluation)
#input = web of trees, user and test_set, max_ratings to receive, max queries to receive
#output = list with queried items
def readWebOfTreesModified(WOT,user,df,max_ratings,max_queries):
    items = []
    ratings = []
    visited_trees = []
    #initiate root
    currentNode = WOT.base.root
    nextNode = currentNode
    aqRatings = 0
    #walk through web_of_trees
    while (len(items)!= max_queries) and (aqRatings != max_ratings):
        #get item of current node and append to list
        item = currentNode.item
    #check if item has already been queried
        #find rating of already queried item
        if len(ratings)!=0 and item in items:
            rating = ratings[items.index(item)]
        #check if user has rated item
        else:
            #append new queried item
            items.append(item)
            #find rating of given user for item of node
            rating = list(df.loc[(df['item']==item) & (df['user']==user), 'rating'])
            if len(rating)==0:      #unknown
                rating = None
            else:                   #rated
                rating = rating[0]
                aqRatings+=1
            ratings.append(rating)
        #check which child needs to be selected based on fact that rating is in interval
        for child in currentNode.children:
            if child==None:
                continue
            if rating in child.chosenInterval:
                nextNode = child
        #if path stops because child is None
        if nextNode == currentNode:
            #take random tree out of web
            partitionNr = np.random.randint(0,WOT.dfSplits)
            #warning if to much trees have been visited
            if len(visited_trees) > WOT.dfSplits*0.9:
                print('Too much trees visited')
                break
            #find partition of tree that has not been visited yet
            while partitionNr in visited_trees:
                partitionNr = np.random.randint(0,WOT.dfSplits)
            visited_trees.append(partitionNr)
            currentNode = (WOT.subTrees[partitionNr]).root
        else:
            currentNode = nextNode
    #return items
    return items    