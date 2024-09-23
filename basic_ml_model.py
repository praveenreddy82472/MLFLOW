import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn


from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score

from sklearn.model_selection import train_test_split

import argparse

def getdata():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        #read the data as df
        df = pd.read_csv(url,sep=";")
        return df
    except Exception as e:
        raise e
def evaluate(y_test,y_pred,pred_prob):
    '''mae=mean_absolute_error(y_test,y_pred)
    mse= mean_squared_error(y_test,y_pred)
    r2Score= r2_score(y_test,y_pred)'''
    
    accuracy = accuracy_score(y_test,y_pred)
    roc_auc= roc_auc_score(y_test,pred_prob,multi_class='ovr')
    return accuracy,roc_auc

    

def main(n_estimators,max_depth):
    df = getdata()
    train,test = train_test_split(df)
     # Prepare the feature and target datasets
    X_train = train.drop('quality', axis=1)
    X_test = test.drop('quality', axis=1)
    
    y_train = train[["quality"]]
    y_test = test[["quality"]]  # Change to use 'test' here
    
    print(X_train.shape)
    print(X_test.shape) 
    print(y_train.shape)
    print(y_test.shape)  
    '''lr = ElasticNet()
    lr.fit( X_train, y_train)
    y_pred = lr.predict(X_test)'''

    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth) #evaluate the model
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    
    pred_prob=rf.predict_proba(X_test)
    
    accuracy,roc = evaluate(y_test, y_pred,pred_prob)
    print("Accuracy:",accuracy)
    print("Roc_Auc:",roc)
        
    

if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators", "-n", default=50, type=int)
    args.add_argument("--max_depth", "-m", default=5, type=int)
    parse_args=args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e
    