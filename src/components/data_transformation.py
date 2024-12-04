import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            categorical_columns=['sex', 'smoker', 'region']
            numerical_columns=['age', 'bmi', 'children']
            sex_map=['female','male']
            smoker_map=['yes','no']
            direction_map=['northeast','southeast','northwest','southwest']
            logging.info('Pipeline has started')

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                       ('OrdinalEncoder',OrdinalEncoder(categories=[sex_map,smoker_map,direction_map])),
                       ('scaler',StandardScaler())
                       ]
            )
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)

    def initiate_data_tarnsformation(self,train_path,test_path):
        try:
            logging.info('Data transformastion statrts')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Read train and test data')
            logging.info(f"Train DataFrame head\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head:\n{test_df.head().to_string()}")
            logging.info("starting Pipeline")
            preprocessor_obj=self.get_data_transformation_object()
            target_column='expenses'
            drop_column=[target_column]
            logging.info('Splitting train data as input target')
            input_column_train_df=train_df.drop(columns=drop_column,axis=1)
            target_column_train_df=train_df[target_column]

            input_column_test_df=test_df.drop(columns=drop_column,axis=1)
            target_column_test_df=test_df[target_column]

            
            input_column_train_arr=preprocessor_obj.fit_transform(input_column_train_df)
            input_column_test_arr=preprocessor_obj.transform(input_column_test_df)
            
            logging.info("Pipeline is Completed")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj
                        )
            
            logging.info("Data Transformed")

            return(
                input_column_train_arr,
                input_column_test_arr,
                target_column_train_df,
                target_column_test_df,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error occured in initiate DataTransformation")
            raise CustomException(e,sys)

