from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pickle
import pandas as pd
import numpy as np

class Randomforest(object):
    
    def __init__(self, dataframe):
        self.data = dataframe

    def saveModel(self, clf, filename):
        self.clf = clf
        self.filename = filename
        joblib.dump(clf, filename)
        
    def loadModel(self, filename):
        self.filename = filename
        classifier = joblib.load(filename)
        return classifier

    def run(self, predict=True):
        filename = "model"
        if not predict:
            
            labels = self.data['Result']
            labels = pd.get_dummies(pd.Series(labels))

            features = self.data
            del features['Result']

            train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 42)
            
            clf = RandomForestClassifier(n_estimators=100,max_depth=8)
            clf.fit(train_features, train_labels)

            file = "file.pkl"
            self.saveModel(clf, filename)
            train_pred = clf.predict(train_features)
            print("Train Accuracy is = ", accuracy_score(train_pred, train_labels))

            test_pred = clf.predict(test_features)
            print("Test Accuracy is = ", accuracy_score(test_pred, test_labels))
            
        else:
            team_dict = {}
            temp_file = open('team_dict', 'rb')
            team_dict = pickle.load(temp_file)
            temp_file.close()

            home_team =  self.data.iloc[0]['Team']
            away_team = self.data.iloc[0]['Opponent Team']
            
            home, away = '', ''
            for key in team_dict.keys():
                if team_dict[key] == home_team:
                    home = key
                if team_dict[key] == away_team:
                    away = key
            clf = self.loadModel(filename)
            test_pred = clf.predict(self.data)

            print ("Predicted value is = ", test_pred)
            if (test_pred[0][0] == 1.0):
                print(home, "Team Won!")
            else:
                print(away, "Team Won!")


