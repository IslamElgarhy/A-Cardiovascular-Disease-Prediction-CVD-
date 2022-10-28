import numpy as np
from google.colab import files
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics




upload = files.upload()
df = pd.read_csv('cardio_train.csv', delimiter=';')

df['age_year'] = df['age']/365
df['smoke'] = df['smoke'] + 1
df['alco'] = df['alco'] + 1
df['active'] = df['active'] + 1

temp_cardop = df['cardio']
df.drop(['cardio'], axis=1, inplace=True)

df['bmi'] = df['weight']/(np.power(df['height']/100, 2))
df['ap_dif'] = df['ap_hi']-df['ap_lo']

#drop_criteria_bmi = df[(df['bmi'] > 24.9) | (df['bmi'] < 18.5)].index

drop_criteria_bmi = df[df['bmi'] > 150].index
df.drop(drop_criteria_bmi, inplace = True)
drop_criteria_aphi = df[(df['ap_hi'] > 210) | (df['ap_hi'] < 60)].index
drop_criteria_aplo = df[(df['ap_lo'] > 140) | (df['ap_lo'] < 30)].index

drop_criteria_ap = df[df['ap_lo'] > df['ap_hi']].index 

## number of records to be removed
drop_criteria = drop_criteria_aphi.union(drop_criteria_aplo)
drop_criteria.union(drop_criteria_ap)
df.drop(drop_criteria, inplace = True)

df['ap_dif'] = df['ap_hi']-df['ap_lo']
df['cardio'] = temp_cardop

df.drop(['id'], axis=1, inplace=True)
df.drop(['age'], axis=1, inplace=True)
df.drop(['gluc'], axis=1, inplace=True)
df.drop(['gender'], axis=1, inplace=True)
df.drop(['active'], axis=1, inplace=True)
df.drop(['smoke'], axis=1, inplace=True)
df.drop(['alco'], axis=1, inplace=True)
#df.drop(['cholesterol'], axis=1, inplace=True)


#df.drop(['bmi'], axis=1, inplace=True)
#df.drop(['ap_dif'], axis=1, inplace=True)

#df.drop(['ap_lo'], axis=1, inplace=True)
#df.drop(['ap_hi'], axis=1, inplace=True)
#df.drop(['weight'], axis=1, inplace=True)
#df.drop(['height'], axis=1, inplace=True)


y = np.array(df['cardio'])
X = np.array(df.drop(['cardio'], 1))
from sklearn.preprocessing import StandardScaler

#age_year, bmi,api_hi,weight,height,ap_dif,ap_lo,cholesterol
trans = StandardScaler()
X = trans.fit_transform(X)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)


# SVM
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#clf.fit(X_train, y_train)
#clf = SVC(kernel='sigmoid') # Linear Kernel
#clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)

print("Accurac _ Training:",acc_linear_svc)
#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("F1Score:",metrics.f1_score(y_test, y_pred))


TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()

print('(tn, fp, fn, tp)')
print((TN, FP, FN, TP))



Sensitivity=(TP/(TP+FN))*100
print("Sensitivity = ", "{:.2f}".format(Sensitivity))


Precision=(TP/(TP+FP))*100
print("Precision = ", "{:.2f}".format(Precision))


Specificity=(TN/(TN+FP))*100
print("Specificity = ", "{:.2f}".format(Specificity))

#Printing the false alarm or false positive rate
FA=(FP/(FP+TN))*100
print("False Alarm = ", "{:.2f}".format(FA))