#!/usr/bin/python 
from __future__ import division
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()   # import some data to play with                                                                                            
iris_X = iris.data            # a copy of iris's list of values (Sepal/Petal Lengths,Widths) in data array                                               
iris_y = iris.target          # a copy of iris's list of values (0,1,2 for the 3 classifiers) in target array                                            
np.unique(iris_y)             # returns sorted unique values of iris_y (target array)                                                                    

# Split iris data in train and test data                                                                                                                 
# A random permutation, to split the data randomly                                                                                                       
#np.random.seed(0)              #Random seed initializing the pseudo ramdom # generator (see Randomstate)                                                
#indices = np.random.permutation(len(iris_X)) #len(iris_X) = 150                                                                                         
#iris_X_train = iris_X[indices[:-10]]  #set training data as first 140 values                                                                            
#iris_y_train = iris_y[indices[:-10]]  #set training target as first 140 values                                                                          
#iris_X_test = iris_X[indices[-10:]]   #set test data as last 10 values                                                                                  
#iris_y_test = iris_y[indices[-10:]]   #set test target as last 10 values                                                                                

#Alternative form of indicies used below                                                                                                                
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.10, random_state=0)                           
# Create and fit a nearest-neighbor classifier                                                                                                           
#knn = KNeighborsClassifier()            #assigns the KNN classifier to a variable                                                                       
#knn.fit(iris_X_train, iris_y_train)     #fit fits the model of X on y                                                                                   
#prediction = knn.predict(iris_X_test)   #predict given unlabeled observations X returns the predicted labels y                                          

k_val =[5,10,15]
def tester(k_array):
    for i in k_array:
        knn = KNeighborsClassifier(n_neighbors = i) #k_array)      #assigns the KNN classifier using current kval to a variable                          
        iris_X_train, iris_X_test, iris_y_train, iris_y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.40, random_state=0)
        knn.fit(iris_X_train, iris_y_train)
        prediction = knn.predict(iris_X_test)
        scores = cross_validation.cross_val_score(knn, iris_X_test, iris_y_test, cv=10)
        print scores, prediction    
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

tester(k_val)
