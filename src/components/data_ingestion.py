import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import mongo_data
import os
import sys
from dataclasses import dataclass
from src.utils import unnecessary
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion starts')

        try:
            data=mongo_data('intern','table')
            logging.info("Dataset readed")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df=data.fetch()
            df=unnecessary(df)

            x=df.drop(labels=['expenses'],axis=1)
            y=df[['expenses']]

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Train test split')

            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=21)
            logging.info(f'x_train DataFrame Head:\n{x_train.head().to_string()}')

            train_set=pd.concat([x_train,y_train],axis=1)
            test_set=pd.concat([x_test,y_test],axis=1)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion')

            raise CustomException(e,sys)

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformer=DataTransformation()
    input_column_train_arr,input_column_test_arr,target_column_train_df,target_column_test_df,preprocessor_obj_path=data_transformer.initiate_data_tarnsformation("artifacts/train.csv","artifacts/test.csv")
    