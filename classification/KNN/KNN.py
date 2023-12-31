import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class KNN:
    def __init__(self,path):
        self.path = path
        self.df =pd.read_csv(self.path)
        self.knn = KNeighborsClassifier(n_neighbors=5)# default 5
        self.kns = KNeighborsClassifier(n_neighbors=11)
        #print(self.df)

    def split_data(self):
        try:
            # converting categorical to numeric
            self.df['diagnosis'].unique()
            self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B': 0}).astype(int)
            # split them into dependent and independent variable
            x = self.df.iloc[:, 1:]
            y = self.df.iloc[:, 0]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            print(f'error in main:{e.__str__()}')

    def modeling_data(self,x_train, x_test, y_train, y_test):
        try:
            self.knn.fit(x_train, y_train) # at default k value =5
            print(f'the value k=5 Training accuracy :{self.knn.score(x_train, y_train)} and testing accuracy:{self.knn.score(x_test, y_test)}')
            y_train_predit = self.knn.predict(x_train)
            y_test_predit = self.knn.predict(x_test)
            # Testing data for k = 5
            print(f'the Training confusion_matrix of data value  k=5 :{confusion_matrix(y_train,y_train_predit)}')
            print(f'the testing confusion_matrix of data value k=5:{confusion_matrix(y_test,y_test_predit)}')
            print(f'the Training classification_report of data value k=5:{classification_report(y_test, y_test_predit)}')
            print(f'the testing classification_report of data value k=5:{classification_report(y_test, y_test_predit)}')
        except Exception as e:
            print(f'error in main:{e.__str__()}')

    def finding_k_value(self, x_train, x_test, y_train, y_test):
        try:
            # finding best k value and k value should be odd only
            k = np.arange(3, 50, 2)
            Train_accuracy = np.empty(len(k))
            Test_accuracy = np.empty(len(k))
            for i, j in enumerate(k):
                self.knns = KNeighborsClassifier(n_neighbors=j)
                self.knns.fit(x_train, y_train)
                Train_accuracy[i] = self.knns.score(x_train, y_train)
                Test_accuracy[i] = self.knns.score(x_test, y_test)
                print(f'Train accuracy{j}={Train_accuracy[i]} and test accuracy{j}={Test_accuracy[i]}')
            print(max(Test_accuracy))
            a = np.where(Test_accuracy == max(Test_accuracy))
            print(f'Indices of Test_accuracy:{a}')

        except Exception as e:
            print(f'error in main:{e.__str__()}')

    def best_k_value(self, x_train, x_test, y_train, y_test):
        try:
            self.kns.fit(x_train, y_train) # got best k value - 11
            print(f'the value k=11 Training accuracy :{self.kns.score(x_train, y_train)} and testing accuracy:{self.kns.score(x_test, y_test)}')
            y_train_predits = self.kns.predict(x_train)
            y_test_predits = self.kns.predict(x_test)
            print(f'the Training confusion_matrix of data value  k=11 :{confusion_matrix(y_train, y_train_predits)}')
            print(f'the testing confusion_matrix of data value k=11:{confusion_matrix(y_test, y_test_predits)}')
            print(f'the Training classification_report of data value k=11:{classification_report(y_train, y_train_predits)}')
            print(f'the testing classification_report of data value k=11:{classification_report(y_test, y_test_predits)}')
        except Exception as e:
            print(f'error in main:{e.__str__()}')

    def dataprocessing(self):
        try:
            self.df.describe()
            self.df.isnull().sum()
            print(f'the missing value:{self.df.isna().sum().sum()}')
            self.df = self.df.drop(['Unnamed: 32', 'id'], axis=1)
            print(f'number of observations:{self.df.shape[0]} and feature:{self.df.shape[1]}')
            x_train, x_test, y_train, y_test = self.split_data()
            print(f'X_train:{x_train.shape} and Y_train:{y_train.shape}')
            print(f'X_test:{x_test.shape} and Y_test:{y_test.shape}')
            self.modeling_data(x_train, x_test, y_train, y_test)
            self.finding_k_value(x_train, x_test, y_train, y_test)
            self.best_k_value(x_train, x_test, y_train, y_test)

        except Exception as e:
            print(f'error in main:{e.__str__()}')


if __name__=='__main__':
    obj=KNN('T:/pycharm/ML/regression/breast-cancer.csv')# load the dataset
    obj.dataprocessing()
