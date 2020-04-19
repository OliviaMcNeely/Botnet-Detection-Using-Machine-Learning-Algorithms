# Botnet-Detection-Using-Machine-Learning-Algorithms

This project aims to detect Botnet traffic from the CTU-13 dataset using single and ensemble Maching Learning algorithms.

The training and test sets have been created from the 12 different scenarios of CTU-13 using the CreateTrainingPickle.py and CreateTestPickle.py scripts which can be found under src.

To run the tests on the single algorithms K-Nearest Neighbour, Decision Tree and Naive Bayes run the script called RunBotnetDetector.py on the command line.

To run the tests on the ensemble algorithms Random Forest, AdaBoost and Gradient Boosting run the script call RunEnsembleBotnetDetector.py on the command line.
