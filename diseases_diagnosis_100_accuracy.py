

import numpy as np 
import pandas as pd 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data1 = pd.read_csv('/content/dataset.csv')
data2 = pd.read_csv('/content/symptom_precaution.csv')
data3 = pd.read_csv('/content/Symptom-severity.csv')
data4 = pd.read_csv('/content/symptom_Description.csv')

data4.loc[16,'Disease'] = 'Dimorphic hemmorhoids(piles)'
data3.loc[102,'Symptom'] = '_patches'

X = data1.iloc[:,1:]
y = data1.iloc[:,0]


def combine(symptoms_list):
    symptoms_list = [x for x in list(symptoms_list) if  isinstance(x, str)]
    return ' '.join(symptoms_list)

X['symp'] = [combine(x) for x in  X.values]

XX = X['symp']

X = X['symp']

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X = X.toarray()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
yy = le.fit_transform(y)



from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X, yy , random_state=44 , test_size=0.2 )

from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import  LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

RND_model = RandomForestClassifier(random_state=44 , n_estimators=100)
ADA_model = AdaBoostClassifier(random_state=44, n_estimators=100)
BAG_model = BaggingClassifier(random_state=44, n_estimators=100)
DST_model = DecisionTreeClassifier(random_state=44)
SVC_model = SVC(random_state=44)
LOG_model = LogisticRegression(random_state=44)
MUL_model = MultinomialNB()
GUS_model = GaussianNB()
KNN_model = KNeighborsClassifier()

from sklearn.metrics import accuracy_score , f1_score

def model_metrics(model , X_train,X_test,y_train,y_test):
    
    #####MODEL##########
    model = model.fit(X_train , y_train)
    y_tests_preds = model.predict(X_test)
    y_train_preds = model.predict(X_train)
    
    ######Accurac#######
    tests_acc =  accuracy_score(y_test , y_tests_preds)
    train_acc =  accuracy_score(y_train , y_train_preds)
    
    ######F1_Score######
    tests_F1 =  f1_score(y_test , y_tests_preds , average='micro')
    train_F1 =  f1_score(y_train , y_train_preds , average='micro')
    
    ####################
    print(f'[!] {model.__class__.__name__}')
    print(f'[+] TESTS ACCURACY : {tests_acc}')
    print(f'[+] TRAIN ACCURACY : {train_acc}')
    print(f'[+] TESTS F1_SCORE : {tests_F1}')
    print(f'[+] TRAIN F1_SCORE : {train_F1}')

model_metrics(RND_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(ADA_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(BAG_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(DST_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(SVC_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(LOG_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(MUL_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(GUS_model,X_train,X_test,y_train,y_test)
print('+'*30)
model_metrics(KNN_model,X_train,X_test,y_train,y_test)

RND_model = RandomForestClassifier(random_state=44 , n_estimators=100)
RND_model.fit(X_test , y_test)

symptoms = ['abdominal_pain acidity anxiety']
test = vectorizer.transform(symptoms).toarray()
y = RND_model.predict(test)

disease = le.classes_[y]


data2.set_index('Disease' , inplace=True)
data4.set_index('Disease' , inplace=True)

disease_info = data4.loc[disease[0]].values[0]

precautions = ', '.join(data2.loc[disease[0]].values)


n_data = pd.DataFrame()
n_data['Disease'] = data1['Disease']
n_data['symp'] = XX

def diagnosis(symptoms):
    test = vectorizer.transform([' '.join(symptoms)]).toarray()
    y = RND_model.predict_proba(test)

    results = list(y[0])
    results.sort()
    most_common = list(dict.fromkeys(results[::-1][:5]))
    diseases_ = []
    proba_disease = []
    for i in most_common:
        index = [indx for indx , v in enumerate(list(y[0])) if v == i ]
        for indx in index:
            disease = le.classes_[indx]
            proba = round(i*100,3)
            diseases_.append(disease)
            proba_disease.append(proba)
    
    return diseases_ , proba_disease

def reults(diseases_ , proba_disease):
    dis_proba = list(zip(diseases_,proba_disease))
    for disease,proba in dis_proba[:4]:
        print(f'[+] PREDICTED DISEASE ({proba}%) : ' , disease)
        try:
            info = data4[data4.index == disease.strip()]['Description'].values[0]
        except:
            info = data4[data4.index == disease]['Description'].values[0]
        print('[!] ABOUT DISEASE : ' , info)

        precautions = ', '.join([x for x in data2.loc[disease].values if isinstance(x , str)])
        print('[X] PRECAUTIONS : ',precautions)

        print('\n')

import collections
def next_questions(diseases_ ,symptoms , answers):
    xz = n_data[n_data['Disease'].isin(diseases_)]
    vectorizer2 = CountVectorizer()
    X1 = vectorizer2.fit_transform(xz['symp'])
    featurez = vectorizer2.get_feature_names_out()
    features_list = []
    for i in X1.toarray():
        for indx , xx in enumerate(i):
            if xx:
                features_list.append(featurez[indx])
    
    counts = collections.Counter(features_list)
    new_list = sorted(features_list, key=lambda x: -counts[x])
    sympa = list(dict.fromkeys(new_list))
    for i in symptoms + answers:
        try:
            sympa.remove(i)
        except:
            pass
    
    return sympa

def risk_calc(symptoms):
    serius_degrees = [data3[data3['Symptom'] == symp].values[0][1] for symp in symptoms if symp in data3['Symptom'].values]
    symp_degree = list(zip(symptoms , serius_degrees))
    rsik_score = sum([x for x in serius_degrees])
    most_serious = ', '.join([x[0] for x in symp_degree if x[1] >= 5])
    print('[+] Risk Score :>> ' , rsik_score)

    if rsik_score in range(15,25):
        print('[-] Plase consult doctor')
    elif rsik_score >= 25:
        print('[-] Plase Go to the nearest emergency department')
    elif rsik_score < 15:
        print("[-] Your risk score is low , so you don't need to take any action! Your symptoms are mild and will disappear in a few hours.")


    print(f'[!] Most Serious Symptoms : {most_serious.replace("_" , " ")}')
    print(f'[!] Risk Score for each Symptom : {symp_degree}')



symptoms = ['abdominal_pain' , 'vomiting' , 'internal_itching']
diseases_ , proba_disease = diagnosis(symptoms)
reults(diseases_ , proba_disease)
risk_calc(symptoms)


import pickle
pickle.dump(RND_model , open('model.pkl' , 'wb'))
pickle.dump(vectorizer , open('vectorizer.pkl' , 'wb'))
pickle.dump(le , open('label_encoder.pkl' , 'wb'))
