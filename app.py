from multiprocessing.connection import answer_challenge
from flask import Flask, g
from flask import request, jsonify, render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import sklearn
app = Flask(__name__)
model=pickle.load(open('healthscore.pkl','rb'))
model2=pickle.load(open('inout.pkl','rb'))

##use disease.pkl
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np # linear algebra
import pandas as pd


vectorizer = CountVectorizer()
data1= pd.read_csv('dataset.csv')
data2= pd.read_csv('symptom_precaution.csv')
data3= pd.read_csv('Symptom-severity.csv')
data4= pd.read_csv('symptom_Description.csv')
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

X = vectorizer.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
yy = le.fit_transform(y)



des_dict=''
des_info=''
des_prec=[]

model=pickle.load(open('disease.pkl','rb'))



def reults(diseases_ , proba_disease):
    dis_proba = list(zip(diseases_,proba_disease))
    for disease,proba in dis_proba[:4]:
		#store disease name in list
        print(f'[+] PREDICTED DISEASE ({proba}%) : ' , disease)
        try:
            info = data4[data4.index == disease.strip()]['Description'].values[0]
        except:
            info = data4[data4.index == disease]['Description'].values[0]
        des_info=info


        precautions = ', '.join([x for x in data2.loc[disease].values if isinstance(x , str)])
        des_prec=print('[X] PRECAUTIONS : ',precautions)

        print('\n')

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



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/frontuser')
def frontuser():
	return render_template('user.html')

@app.route('/user', methods=['GET', 'POST'])
def user():
	if request.method == 'POST':
		user = request.form.getlist('user')
		symptoms = user
		test = vectorizer.transform(symptoms).toarray()
		y = model.predict(test)
		z = model.predict_proba(test)
		data2.set_index('Disease' , inplace=True)
		data4.set_index('Disease' , inplace=True)
		disease = le.classes_[y]

		disease_info = data4.loc[disease[0]].values[0]
		precaution = data2.loc[disease[0]].values[0]

		n_data = pd.DataFrame()
		n_data['Disease'] = data1['Disease']
		n_data['symp'] = XX

		results = list(z[0])
		results.sort()
		most_common = list(dict.fromkeys(results[::-1][:5]))
		diseases_ = []
		proba_disease = []
		for i in most_common:
			index = [indx for indx, v in enumerate(list(z[0])) if v == i]
			for indx in index:
				disease = le.classes_[indx]
				proba = round(i*100, 3)
				diseases_.append(disease)
				proba_disease.append(proba)
		
		reults(diseases_ , proba_disease)
		risk_calc(symptoms)
		return redirect(url_for('display', value=des_info))
	return render_template('user.html')
@app.route('/hscore', methods=['GET','POST'])
def hscore():
	global a
	global b
	global c
	global d
	global e
	global f
	if request.method == 'POST':
		a = request.form['a']
		b = request.form['b']
		c = request.form['c']
		d = request.form['d']
		e = request.form['e']
		f = request.form['f']
		df=pd.DataFrame([[a,b,c,d,e,f]],columns=['TEMPF','PULSE',	'RESPR', 'BPSYS',	'BPDIAS',	'POPCT'])
		global prediction
		prediction=model.predict(df)
		answer=''
		if prediction==1 or prediction==0:
			answer='Low risk of rehospitalization'
		elif prediction==2:
			answer='Medium risk of rehospitalization'
		elif prediction==3:
			answer='High risk of rehospitalization'
		return redirect(url_for('display', value=5))

	return render_template('admin1.html')

@app.route('/inout', methods=['GET','POST'])
def inout():
	global a1
	global b1
	global c1
	global d1
	global e1
	global f1
	global g1
	global h1
	global i1
	global j1
	if request.method == 'POST':
		a1 = request.form['a1']
		b1 = request.form['b1']
		c1 = request.form['c1']
		d1 = request.form['d1']
		e1 = request.form['e1']
		f1 = request.form['f1']
		g1 = request.form['g1']
		h1 = request.form['h1']
		i1 = request.form['i1']
		j1 = request.form['j1']
		df=pd.DataFrame([[a1, b1, c1, d1, e1, f1, g1, h1, i1, j1]],columns=['HAEMATOCRIT',	'HAEMOGLOBINS',	'ERYTHROCYTE',	'LEUCOCYTE',	'THROMBOCYTE',	'MCH',	'MCHC',	'MCV',	'AGE',	'SEX'])
		prediction=model2.predict(df)
		answer=''
		if prediction == 0:
			answer='Patient will be discharged'
		else:
			answer='Patient will be admitted'
		return redirect(url_for('display', value=answer))
	return render_template('admin2.html')

@app.route('/admin')
def admin():
	return render_template('admin.html')


@app.route('/login')
def login():
	return render_template('login.html')

@app.route('/display/<value>')
def display(value):
	return render_template('display.html', prediction=value)
	
@app.route('/userdisp')
def userdisp():
	return render_template('userdisp.html')

if __name__ == "__main__":
    app.run(debug=True)