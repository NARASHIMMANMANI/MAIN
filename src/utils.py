import pandas as pd
import numpy as np
import pymongo
import csv
from src.exception import CustomException
from src.logger import logging
import sys
import os
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

class mongo_data:
    def __init__(self,db_name,collection_name):
        self.db_name=db_name
        self.collection_name=collection_name
    def fetch(self):
        try:
            logging.info('fetching data from Mongo DB has started')
            client=pymongo.MongoClient("mongodb://localhost:27017/")
            db=client[self.db_name]
            collection=db[self.collection_name]
            cursor=collection.find()
            data_list=[]
            for doc in cursor:
                data_list.append(doc)
            data=pd.DataFrame(data_list)
            logging.info('fetching data from Mongo DB has finished')

            return data
        except Exception as e:
            logging.info("Error occured on Mongodb_data")
            raise CustomException(e,sys)
def unnecessary(data):
    df=data.drop('_id',axis=1)
    return df

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def model_evaluate(x_train,y_train,x_test,y_test,models):
    try:
        report={}

        for model_name,model in models.items():
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            r_square=r2_score(y_test,y_pred)
            mse=mean_squared_error(y_test,y_pred)*100
            report[model_name]=r_square,mse
        return report
    except Exception as e:
        logging.info('Exception occured during model Training')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function')
        raise CustomException(e,sys)