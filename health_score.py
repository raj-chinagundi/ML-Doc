

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report,f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier



fig_X = 10
fig_y = 8
bins = 25
title_size = 20
color = 'b'



score_df = pd.read_csv('/content/Patient Severity Score for Electronic Health Records.csv')
score_df.head()


score_df = score_df.rename(columns={'SCORE ':'SCORE'})


score_df.isnull().sum()



X = score_df.drop('SCORE',axis=1)
y = score_df.SCORE



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=11)



scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

import pickle
with open('scalar.pkl','wb') as f:
    pickle.dump(scalar,f)



model_name = list()
accuracy = list()
models = {
    LogisticRegression(max_iter=500):'Logistic Regression',
    SVC():"Support Vector Machine",
    RandomForestClassifier():'Random Forest'
}
for m in models.keys():
    m.fit(X_train,y_train)
for model,name in models.items():
     print(f"Accuracy Score for {name} is : ",model.score(X_test,y_test)*100,"%")
     model_name.append(name)
     accuracy.append(model.score(X_test,y_test)*100)