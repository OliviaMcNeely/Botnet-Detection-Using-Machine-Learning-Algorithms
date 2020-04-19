'''
By Olivia McNeely
'''

import pickle
import pandas as pd
from sklearn import metrics
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn import svm
        
        
def run():
    #Opening the training data pickle file
    try:
        print("Opening training data...")
        
        trainingdata = pd.read_pickle("../dataset/trainingData.pickle")
        trainData = trainingdata[0]
        trainLabel = trainingdata[1]
            
        print("Training data opened!")
        
    except Exception as e:
        print("Error opening training data file!")
        print(e)
        
        
    #Getting input from the command line to choose with testset to run tests on
    while True:
        try:
            testsetInput = int(input("\nEnter a testset number from 2 to 12: "))
            if 2 <= testsetInput <= 12:
                break
            else:
                raise ValueError()
        except ValueError:
            print("Input must be an integer from 2 to 12.")
    
    
    #Opening the required testset based on input
    try:
        print("\nOpening test data...")
        if testsetInput == 2:
            testingdata = pd.read_pickle("../dataset/testData2.pickle")
        elif testsetInput == 3:
            testingdata = pd.read_pickle("../dataset/testData3.pickle")
        elif testsetInput == 4:
            testingdata = pd.read_pickle("../dataset/testData4.pickle")
        elif testsetInput == 5:
            testingdata = pd.read_pickle("../dataset/testData5.pickle")
        elif testsetInput == 6:
            testingdata = pd.read_pickle("../dataset/testData6.pickle")
        elif testsetInput == 7:
            testingdata = pd.read_pickle("../dataset/testData7.pickle")
        elif testsetInput == 8:
            testingdata = pd.read_pickle("../dataset/testData8.pickle")
        elif testsetInput == 9:
            testingdata = pd.read_pickle("../dataset/testData9.pickle")
        elif testsetInput == 10:
            testingdata = pd.read_pickle("../dataset/testData10.pickle")
        elif testsetInput == 11:
            testingdata = pd.read_pickle("../dataset/testData11.pickle")
        elif testsetInput == 12:
            testingdata = pd.read_pickle("../dataset/testData12.pickle")
            
        testData = testingdata[0] #Data from the testset
        testLabel = testingdata[1] #Labels for the testset
            
        print("Test data opened!")
            
    except Exception as e:
        print("Error opening test data file!")
        print(e)
        
   
    #Getting input from the command line to select with model to run
    try:
        algorithmInput = int(input("\nSelect an algorithm or -1 to quit: \n1.K-Nearest Neighbours\n2.Decision Tree\n3.Naive Bayes\n"))
        
        while algorithmInput != -1:
            
            if algorithmInput < 1 or algorithmInput > 3:
                raise ValueError()
            elif algorithmInput == -1:
                break
            else:
                if algorithmInput == 1:        
                    KNN(trainData, trainLabel, testData, testLabel)
                elif algorithmInput == 2:
                    DT(trainData, trainLabel, testData, testLabel)
                elif algorithmInput == 3:
                    NB(trainData, trainLabel, testData, testLabel)
                    
            algorithmInput = int(input("\nSelect an algorithm or -1 to quit: \n1.K-Nearest Neighbours\n2.Decision Tree\n3.Naive Bayes\n"))
             
    except ValueError:
        print("\nInput must be number from 1-3.")

        
#K-Nearest Neighbours model      
def KNN(trainData, trainLabel, testData, testLabel):
    try:
        print("\nRunning K Nearest Neighbours...")

        clf = KNeighborsClassifier(n_neighbors=10)

        clf = clf.fit(trainData, trainLabel) #Training the data

        prediction = clf.predict(testData) #Predicting the data

        accuracy = metrics.accuracy_score(testLabel, prediction) * 100
        print("Accuracy of K Nearest Neighbour Model: %.2f" % accuracy+' %')
                        
        precision = metrics.precision_score(testLabel, prediction)
        print("Precision of K Nearest Neighbour Model: %.2f" % precision)
                        
        recall = metrics.recall_score(testLabel, prediction)
        print("Recall of K Nearest Neighbour Model: %.2f" % recall)
                        
        fmeasure = 2 * ((precision * recall)/(precision + recall))
        print("F-Measure of K Nearest Neighbour Model: %.2f" % fmeasure)

    except Exception as e:
        print("\nError with KNN model")
        print(e)


#Decision Tree model  
def DT(trainData, trainLabel, testData, testLabel):
    try:
        print("\nRunning Decision Tree...")

        clf = DecisionTreeClassifier()

        clf = clf.fit(trainData, trainLabel) #Training the data

        prediction = clf.predict(testData) #Predicting the data

        accuracy = metrics.accuracy_score(testLabel, prediction) * 100
        print("Accuracy of Decision Tree Model: %.2f" % accuracy+' %')
                        
        precision = metrics.precision_score(testLabel, prediction)
        print("Precision of Decision Tree Model: %.2f" % precision)
                        
        recall = metrics.recall_score(testLabel, prediction)
        print("Recall of Decision Tree Model: %.2f" % recall)
                        
        fmeasure = 2 * ((precision * recall)/(precision + recall))
        print("F-Measure of Decision Tree Model: %.2f" % fmeasure)

    except Exception as e:
        print("\nError with DT model")
        print(e)
        

#Naive Bayes model 
def NB(trainData, trainLabel, testData, testLabel):
    try:
        print("\nRunning Naive Bayes...")

        clf = GaussianNB()

        clf = clf.fit(trainData, trainLabel) #Training the data

        prediction = clf.predict(testData) #Predicting the data

        accuracy = metrics.accuracy_score(testLabel, prediction) * 100
        print("Accuracy of Gaussian Naive Bayes Model: %.2f" % accuracy+' %')
                        
        precision = metrics.precision_score(testLabel, prediction)
        print("Precision of Gaussian Naive Bayes Model: %.2f" % precision)
                    
        recall = metrics.recall_score(testLabel, prediction)
        print("Recall of Gaussian Naive Bayes Model: %.2f" % recall)
                        
        fmeasure = 2 * ((precision * recall)/(precision + recall))
        print("F-Measure of Gaussian Naive Bayes Model: %.2f" % fmeasure)

    except Exception as e:
        print("\nError with NB model")
        print(e)

        
if __name__ == "__main__":
    run()