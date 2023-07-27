import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
class SLR:
    def __init__(self,path):
        self.path=path
        self.df = pd.read_csv(self.path)
        self.model = LinearRegression()
        #print(self.df)

    def split_data(self):
        try:
            x = self.df['cgpa']
            y = self.df['package']
            x = x.values.reshape(-1, 1)
            x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return x_train, y_train, x_test, y_test

        except Exception as e:
            print(f'error in main:{e.__str__()}')

    def modeling_data(self,x_train,y_train,x_test,y_test):
        try:
            self.model.fit(x_train,y_train)
            print(f'Training accuracy model:{self.model.score(x_train,y_train)}')
            print(f'Testing accuracy model:{self.model.score(x_test, y_test)}')
            print(f'LOSS Testing accuracy model:{1-(self.model.score(x_test, y_test))}')
            print(f'CGPA for 8.0 and package he get:{self.model.predict([[8.0]])}')

        except Exception as e:
            print(f'error in main:{e.__str__()}')


    def simplelinear_data(self):
        try:
            self.df.isnull().sum()
            print(f'missing value:{self.df.isna().sum().sum()}')
            print(f'number of observation:{self.df.shape[0]} and feature:{self.df.shape[1]}')
            x_train, y_train, x_test, y_test = self.split_data()
            Training_data=pd.DataFrame({'X_train':x_train.flatten(),'Y_Train':y_train})
            print(f'Training_Data:{Training_data}')
            self.modeling_data(x_train,y_train,x_test,y_test)


        except Exception as e:
            print(f'error in main:{e.__str__()}')


if __name__== '__main__':
    obj = SLR('T:/pycharm/ML/regression/placement.csv')
    obj.simplelinear_data()
