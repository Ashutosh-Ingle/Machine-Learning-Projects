import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
    # step 1 : Load data 

    titanic_data = pd.read_csv("TitanicDataset.csv")
    print("First 5 entries from loaded dataset ")
    print(titanic_data.head())

    print("Number of passengers are "+str(len(titanic_data)))

    # step 2 : Analyze data
    print("Visualisation : Survived and non survived passengers")
    figure()
    target = "Survived"

    countplot(data= titanic_data, x = target).set_title("Ashutosh ML : Survived and non-Survived passengers")
    show()
    
    print("Visualisation : Survived and non survived passengers based on Gender")
    figure()
    target = "Survived"

    countplot(data= titanic_data, x= target, hue = "Sex").set_title("Ashutosh ML :  Survived and non-Survived based on gender")
    show()

    print("Visualisation : Survived and non survived passengers based on passenger class")
    figure()
    target = "Survived"

    countplot(data = titanic_data,x= target, hue = 'Pclass').set_title("Ashutosh ML : Survived and non survived based on passenger class")
    show()

    print("Visualisation : Survived and non survived passengers based Age")
    figure()
    titanic_data["Age"].plothist().set_title("Ashutosh ML : Survived and non Survived based on Age")
    show()

    print("Visualisation : Survived and non survived passengers based on Fare")
    figure()
    titanic_data("Fare").plot.hist().set_title("Ashutosh ML : Survived and non survived based on Fare")
    show()

    # step 3 : Data cleaning
    titanic_data.drop("Zero", axis = 1, inplace = True)

    print("First 5 entries loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("Values of Sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first = True)
    print(Sex.head(5))

    print("Values of Pclass Column after removinf one field ")
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first = True)
    print(Pclass.head(5))

    print("Values of data set after concatenating new columns")
    titanic_data = pd.concat((titanic_data,Sex,Pclass), axis = 1)
    print(titanic_data.head(5))

    print("Values of data set after removijng irrelevent columns")
    titanic_data.drop(['Sex','sibsp','Parch','Embarked'],axis = 1, inplace = True)
    print(titanic_data.head(5))

    x = titanic_data.drop("Survived",axis = 1)
    y = titanic_data["Survived"]

    #step 4 : Data training 
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size = 0.5)

    logmodel = LogisticRegression()
    
    logmodel.fit(xtrain, ytrain)

    #step 4 : Data testing 
    prediction = logmodel.predict(xtest)

    #step 5 : Calculate Accuracy
    print("Classification report of logistic Regression is: ")
    print(classification_report(ytest,prediction))

    print("Confusion matrix of logistic Regression is: ")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of Logistice Regression is: ")
    print(accuracy_score(ytest,prediction))

def main():
    print("Supervised Machine Learning")

    print("Logistic Regression on Titanic data set")

    TitanicLogistic()

if __name__ == "__main__":
    main()







