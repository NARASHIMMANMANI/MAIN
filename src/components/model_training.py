import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.utils import save_object
from src.utils import model_evaluate
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_data_input,test_data_input,train_data_target,test_data_target):
        try:
            logging.info("Splitting Dependent and Independent Variable")
            X_train,y_train,X_test,y_test=train_data_input,train_data_target,test_data_input,test_data_target
            models={
            'LinearRegression':LinearRegression(),
            'Ridge':Ridge(),
            'Lasso':Lasso(),
            #'Polynomial_regression':PolynomialFeatures(),
            'Support vector Machine':SVR(),
            'DTR':DecisionTreeRegressor(),
            'RandomForest':RandomForestRegressor(),
            'Neighbors':KNeighborsRegressor(),
            'Gaussion':GaussianProcessRegressor(),
            'Neural_network':MLPRegressor(),
            'BayesianRidge':BayesianRidge()
            }

            model_report:dict=model_evaluate(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n==================================')
            logging.info(f"Model Report :{model_report}")
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            print(f'Best Model Found,Model Name :{best_model_name},accuracy:{best_model_score}')
            print('\n===============================================')
            logging.info(f'Best Model Found,Model Name:{best_model_name},accuracy:{best_model_score}')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info('Excepton occured at model Training')
            raise CustomException(e,sys)