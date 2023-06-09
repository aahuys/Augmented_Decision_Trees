__author__ = 'Aaron Huys'
#read in data
import numpy as np
import pandas as pd

############################### 25M dataset ###############################
#function for getting the ratings data
#input: None
#output: Pandas datafram
#first row of the array is the header (userId,movieId,rating,timestamp)
def InitRatingsData25M():
    data = pd.read_csv("./dataset25M/ratings.csv")
    data.drop('timestamp', axis=1, inplace=True)
    print('Dataset ratings.csv loaded')
    return data

#function for getting the ratings data
#input: None
#output: Pandas datafram
#first row of the array is the header (movieId,title,genres)
#genres contained: * Action* Adventure* Animation* Children's*
#Comedy* Crime* Documentary* Drama* Fantasy* Film-Noir* Horror* 
#Musical* Mystery* Romance* Sci-Fi* Thriller* War* Western * 
#(no genres listed)
def InitMoviesData25M():
    data = pd.read_csv("./dataset25M/movies.csv")
    print('Dataset movies.csv loaded')
    return data

#function for getting the tags data
#input: None
#output: Pandas datafram
#first row of the array is the header (userId,movieId,tag,timestamp)
def InitTagsData25M():
    data = pd.read_csv("./dataset25M/tags.csv")
    print('Dataset tags.csv loaded')
    return data

#function for getting the links data
#input: None
#output: Pandas datafram
#first row of the array is the header (movieId,imdbId,tmdbId)
def InitLinksData25M():
    data = pd.read_csv("./dataset25M/links.csv")
    print('Dataset links.csv loaded')
    return data

#function for getting the ratings data put only a subset where a 
#         fraction of the users is used.
#input: integer split
#output: Pandas datafram
#first row of the array is the header (userId,movieId,rating,timestamp)
def InitRatingsData25MSmall(fraction):
    #results in same split of dataset for different tries
    rng = np.random.RandomState(1)
    data = pd.read_csv("./dataset25M/ratings.csv")
    data.drop('timestamp', axis=1, inplace=True)
    all_users = data['userId'].unique()
    #create subset of users
    sub_users = rng.choice(all_users,
                           np.shape(all_users)[0]//fraction,
                           replace=False)
    #filter out users
    data = data[data['userId'].isin(sub_users)]
    print('Dataset ratings.csv loaded')
    return data

#function for getting the final data subset
#input: integer split
#output: Pandas datafram
#first row of the array is the header (userId,movieId,rating,timestamp)
def InitRatingsSubSet(fraction):
    #get user based split of dataset
    df = InitRatingsData25MSmall(fraction)
    #set seed
    rng = np.random.RandomState(1)
    #counts of movie occurences
    counts = df['movieId'].value_counts()
    #hard coded manipulation
    fractions = [1]+[0.7/(i**2) for i in range(1,11)]
    tot_removables = np.array([0])
    #change items with less than 10 ratings
    for i in range(1,11):
        #get all itemids with i ratings
        items = counts[counts==i].index.to_numpy()
        #find itemids to filter out
        size = np.shape(items)[0]*fractions[i]
        removables = rng.choice(items,int(size),replace=False)
        #add to all item ids that need to be removed
        tot_removables = np.concatenate((tot_removables,removables))
    #edit dataset
    df = df[~df['movieId'].isin(tot_removables)]
    return df


############################### 1M dataset ###############################
#function for getting the ratings data
#input: None
#output: Pandas datafram
#first row of the array is the header (userId,movieId,rating,timestamp)
def InitRatingsData1M():
    data = pd.read_csv("./dataset1M/ratings.dat",sep='::',engine='python',header=None)
    data.columns = ['userId','movieId','rating','timestamp']
    print('Dataset ratings.dat loaded')
    return data

#function for getting the ratings data
#input: None
#output: Pandas datafram
#first row of the array is the header (movieId,title,genres)
#genres contained: * Action* Adventure* Animation* Children's* Comedy*
#Crime* Documentary* Drama* Fantasy* Film-Noir* Horror* Musical* Mystery*
#Romance* Sci-Fi* Thriller* War* Western * (no genres listed)
def InitMoviesData1M():
    data = pd.read_csv("./dataset1M/movies.dat",sep='::',engine='python',header=None)
    print('Dataset movies.dat loaded')
    return data

#function for getting the links data
#input: None
#output: Pandas datafram
#first row of the array is the header (movieId,imdbId,tmdbId)
def InitUsersData1M():
    data = pd.read_csv("./dataset1M/users.dat",sep='::',engine='python',header=None)
    data.columns = ['userId','gender','age','occupation','zip-code']
    print('Dataset users.dat loaded')
    return data

############################### 100K dataset ###############################
#function for getting the ratings data
#input: None
#output: Pandas datafram
#first row of the array is the header (userId,movieId,rating,timestamp)
def InitRatingsData100K():
    data = pd.read_csv("./dataset100K/ratings.csv")
    print('Dataset ratings.csv loaded')
    return data

#function for getting the ratings data
#input: None
#output: Pandas datafram
#first row of the array is the header (movieId,title,genres)
#genres contained: * Action* Adventure* Animation* Children's* Comedy*
#Crime* Documentary* Drama* Fantasy* Film-Noir* Horror* Musical* Mystery*
#Romance* Sci-Fi* Thriller* War* Western * (no genres listed)
def InitMoviesData100K():
    data = pd.read_csv("./dataset100K/movies.csv")
    print('Dataset movies.csv loaded')
    return data