import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')

class   LOGR:
    def __init__(self,path):
        self.path=path
        self.nb=pd.read_csv(self.path)
        self.log=LogisticRegression()

    def split_data(self):
        try:
            # convert categorical to numerical
            self.nb['diagnosis'].unique()
            self.nb['diagnosis'] = self.nb['diagnosis'].map({'M': 1, 'B': 0}).astype(int)
            x = self.nb.iloc[:, 1:]
            y = self.nb.iloc[:, 0]
            print(x)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            print(f'error in main:{e.__str__()}')

    def modeling_data(self,x_train, x_test, y_train, y_test):
        try:
            self.log.fit(x_train, y_train)
            y_train_predict = self.log.predict(x_train)
            y_test_predict = self.log.predict(x_test)
            print(f'the Training accuracy:{accuracy_score(y_train,y_train_predict)}')
            print(f'the Testing accuracy:{accuracy_score(y_test, y_test_predict)}')
            # Validation Report
            print(f'the Training confusion:{confusion_matrix(y_train,y_train_predict)}')
            print(f'the Testing confusion:{confusion_matrix(y_test, y_test_predict)}')
            print(f'the Training classification report:{classification_report(y_train, y_train_predict)}')
            print(f'the Testing classification report:{classification_report(y_test, y_test_predict)}')

        except Exception as e:
            print(f'error in main:{e.__str__()}')


    def dataprocessing(self):
        try:
            self.nb.isnull().sum()# checking null values
            print(f'Missing null values:{self.nb.isna().sum()}')
            self.nb=self.nb.drop(['Unnamed: 32','id'],axis=1)
            print(f'OBESERVATION AND FEATURE:{self.nb.shape[0]} and {self.nb.shape[1]}')
            x_train, x_test, y_train, y_test=self.split_data()
            self.modeling_data(x_train, x_test, y_train, y_test)


        except Exception as e:
            print(f'error in main:{e.__str__()}')



if __name__=='__main__':
    obj=LOGR("T:/pycharm/ML/supervisied_model/breast-cancer.csv")# load the dataset
    obj.dataprocessing()