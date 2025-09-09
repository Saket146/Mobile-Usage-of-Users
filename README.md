# Mobile-Usage-of-Users

Welcome to my project !!

This is a simple machine learning project which will classify people into 5 groups(Very low = 1, Low = 2, Moderate = 3, High = 4, Very high = 5 ) based upon their usage of phones. The dataset contains 700 data members. All thanks to kaggle for this dataset.

Dimensionality reduction technique : Principle Component Analysis. Techniques used for model development : Xgboost. For better results of model and to prevent overfitting of data I have used k-fold cross validation.

Let's see in detail.

#Importing important libraries
In this section I have imported the following three important libraries:

Pandas as pd
Numpy as np
Matplotlib as plt
Importing the dataset
In this section I have imported the dataset. The variable X has the columns neccesary for making a prediction such as "App Usage time", "Screen ON Time", "Battery Drained", "Data Used". The other columns does not affect the prediction. So there is no need to include them. The variable y has the group in which people are classified based upon the data in variable X. The Xgboost model expects the data in the dependent variable(y) to have the starting value as 0. So i have decremented all the values in y by 1.

#Splitting the data into training set and test set
Our data need to be split into training data and testing data before start of model building. We train the model on the training data and test our model using the testing data to detect any bugs or errors in our model. Usually the testing data consists of 20% of whole data. X_train, y_train consists of training data. X_test and y_test consists of testing data.

#Feature Scalling
The features(data in x) are scalled using the standardizing technique which is given as

X' = (X - Xmin)/(Xmax - Xmin)

It is used to prevent some of the features in dataset to get dominated by other features and the model do not even consider those features.

#Principal Component Analysis
It finds the co-relaton between the variables and if there is a strong relation it will reduce the variables In this way it will reduce the dimensions and make the model easy to learn.

#Building the model
I trained my model on Xgboost. To use this we have to import XGBClassifier from Xgboost module and create an instance of this class as "Classifier". Fit the X_train, y_train values so that model gets trained on Xgboost using training values.

Create a confusion matrix and accuracy score for our testing data which will describe how our model is responding to new values. Confusion matrix shows the exact predictions of our model by showing the ones which were predicted incorrect and which were predicted correct. "confusion_matrix" and "accuracy_score" are imported from the sklearn library. Introduce a new variable "y_pred" which will take the prediction of dependent variable (To which group the user belong). Display the confusion matrix and accuracy for our testing data. Here we see that 3 predictions of group 2 were predicted incorrectly and we got a good accuracy of 97.8 %.

But sometimes we may get lucky for getting this better accuracy. So we train the data by splitting the training data into multiple parts and store the accuracies of every part in a list. This is called as "k - fold cross validation" where k represents the number of parts the training data is divided. We then caluculate the mean of the accuracies and also mean deviation which will improve our model and make it even more robust.

k-fold cross validation method can be implemented by importing the "cross_val_score" method from the sklean library. We then append all the resulting accuracies of every portion in a list called "accurcies". The cross_val_score requires few parameters that need to be passed. They are:

estimator:- It represents the variable on which the model is trained. Here it is classifier.
X and y values:- As we are training k-fold cross validation on training set X = X_train, y = y_train
cv:- It represents how many parts are we splitting the training data i.e, the value of k. Here we take cv as 10 for better results.
Then we can see the average accuracy as 98.93 % and standard deviation as 1.18 %. With this we can say that our model is ready for implementation and will give accurate results as given in the sample prediction.
