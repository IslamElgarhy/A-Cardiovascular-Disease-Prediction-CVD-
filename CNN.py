from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers
from keras.utils.np_utils import to_categorical


from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


import numpy as np
from google.colab import files
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
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

#df.drop(['ap_lo'], axis=1, inplace=True)
#df.drop(['ap_hi'], axis=1, inplace=True)
#df.drop(['weight'], axis=1, inplace=True)
#df.drop(['height'], axis=1, inplace=True)

#df.drop(['ap_dif'], axis=1, inplace=True)
#df.drop(['bmi'], axis=1, inplace=True)
#age_year, bmi,api_hi,weight,height,ap_dif,ap_lo,cholesterol

y = np.array(df['cardio'])
X = np.array(df.drop(['cardio'], 1))

trans = StandardScaler()
X = trans.fit_transform(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)


# define a new keras model for binary classification
def create_binary_1Dmodel():
    # create model
    model = Sequential()
    model.add(Conv1D(128, 6, activation="relu",kernel_regularizer=regularizers.l2(0.00001), input_shape=(8,1)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.05))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(2, activation = 'softmax'))
    opt = Adam(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    
    return model

binary_1Dmodel = create_binary_1Dmodel()
print(binary_1Dmodel.summary())


y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)


history=binary_1Dmodel.fit(X_train, y_train, epochs=100, verbose=1, batch_size=10 )


# evaluate the keras model on test data
_, accuracy = binary_1Dmodel.evaluate(X_test, y_test)
print('Accuracy on test data: %.2f' % (accuracy*100))


y_pred1 = binary_1Dmodel.predict(X_test)
y_pred = np.argmax(y_pred1, axis=1)
ytst = np.argmax(y_test, axis=1)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(ytst, y_pred)*100)

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(ytst, y_pred)*100)

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(ytst, y_pred)*100)

# Model Recall: what percentage of positive tuples are labelled as such?
print("F1Score:",metrics.f1_score(ytst, y_pred)*100)


TN, FP, FN, TP = metrics.confusion_matrix(ytst, y_pred).ravel()

print('(tn, fp, fn, tp)')
print((TN, FP, FN, TP))



#Recall
Sensitivity=(TP/(TP+FN))*100
print("Sensitivity = ", "{:.2f}".format(Sensitivity))

Precision=(TP/(TP+FP))*100
print("Precision = ", "{:.2f}".format(Precision))

Specificity=(TN/(TN+FP))*100
print("Specificity = ", "{:.2f}".format(Specificity))

#Printing the false alarm or false positive rate
FA=(FP/(FP+TN))*100
print("False Alarm = ", "{:.2f}".format(FA))