import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tsfel
  
from sklearn.tree import DecisionTreeClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import DeepFool

from art.estimators.classification import SklearnClassifier
from art.estimators.classification import KerasClassifier
from numpy import loadtxt
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D,Dropout,GRU,LSTM,SimpleRNN
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam ,SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras import regularizers
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn.metrics import precision_score , recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from numpy import array
from sklearn.metrics import accuracy_score
from numpy import tensordot
from numpy import argmax
import matplotlib.pyplot as plt
import copy
import statsmodels.tsa.api as smt

def increase_font():
  from IPython.display import Javascript
  display(Javascript('''
  for (rule of document.styleSheets[0].cssRules){
    if (rule.selectorText=='body') {
      rule.style.fontSize = '30px'
      break
    }
  }
  '''))


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testX0,testX1,testX2,testX3,testX4):
  # make predictions
  yhats = []
  yhats.append(members[0].predict(testX0))
  yhats.append(members[1].predict(testX1))
  yhats.append(members[2].predict(testX2))
  yhats.append(members[3].predict(testX3))
  yhats.append(members[4].predict(testX4))

  yhats = array(yhats)
  #print(yhats.shape)
  # weighted sum across ensemble members
  summed = tensordot(yhats, weights, axes=((0),(0)))
  # argmax across classes
  result = argmax(summed, axis=1)
  return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX0,testX1,testX2,testX3,testX4, testy):
	# make prediction
	yhat = ensemble_predictions(members,weights,testX0, testX1,testX2,testX3,testX4)
	# calculate accuracy
	return accuracy_score(testy, yhat)

def display_metric(predictions,ytst, xalgo, xtype):
  #increase_font()
  y_pred_i = np.argmax(predictions, axis=1)
  ytst_i = np.argmax(ytst, axis=1)
  print(xalgo," on ", xtype,"samples:")
  TN, FP, FN, TP = confusion_matrix(ytst_i, y_pred_i).ravel()
  print('Confusion Matrix= (tn=',TN,', fp=',FP,', fn=',FN,', tp=',TP,')')
  FA=(FP/(FP+TN))*100
  print("False Alarm = ", "{:.2f}".format(FA))
  print("Accuracy = ", "{:.2f}".format(metrics.accuracy_score(ytst_i, y_pred_i)*100))
  print("Recall = ", "{:.2f}".format(metrics.recall_score(ytst_i, y_pred_i)*100))
  print("Precision_score = ", "{:.2f}".format(metrics.precision_score(ytst_i, y_pred_i)*100))
  print ("F1_score=", "{:.2f}".format(f1_score(ytst_i, y_pred_i)*100))


def display_Accuracy(predictions,ytst, xalgo, xtype):
  y_pred_i = np.argmax(predictions, axis=1)
  ytst_i = np.argmax(ytst, axis=1)
  print(xalgo," on ", xtype,":", "Accuracy = ", "{:.2f}".format(metrics.accuracy_score(ytst_i, y_pred_i)*100))

# Step 1: Load the dataset
X_train=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/X_train.csv')
Y_train=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/Y_train.csv')
X_test=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/X_test.csv')
Y_test=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/Y_test.csv')

# Step 1: Load the dataset
X_train_attacker=pd.read_csv('drive/MyDrive/myColabData/Data/65Attacker/X_train.csv')
Y_train_attacker=pd.read_csv('drive/MyDrive/myColabData/Data/65Attacker/Y_train.csv')
X_train_h=pd.read_csv('drive/MyDrive/myColabData/Data/65Attacker/X_honest.csv')
X_test_attacker=pd.read_csv('drive/MyDrive/myColabData/Data/65Attacker/X_test.csv')
Y_test_attacker=pd.read_csv('drive/MyDrive/myColabData/Data/65Attacker/Y_test.csv')
X_test_attacker_forEvasion=pd.read_csv('drive/MyDrive/myColabData/Data/65Attacker/X_attack.csv')
Y_test_attacker_forEvasion=pd.read_csv('drive/MyDrive/myColabData/Data/65Attacker/Y_attack.csv')


#acovf
#xx_X_train=[]
#for i in range(46539):
  #xx_X_train.append(smt.stattools.acovf(X_train.iloc[i, 0:48])) #autocovariance upto 20 values
#  x = smt.stattools.acf(X_train.iloc[i, 0:48],unbiased=True,nlags=23)
#  if(np.isnan(x).any()):
#    x = np.zeros(48)
#  else:
#    xx_X_train.append(x)


#cfg = tsfel.get_features_by_domain()
#data_train = []
#for i in range(10):
#  print(i)
#  x_before = X_train.iloc[i, 0:48]
#  x_after = tsfel.time_series_features_extractor(cfg, x_before)
#  data_train.append(x_after)

#x = np.array(data_train).reshape(10,159)
#with open('xtr_Stat_NEW5.csv','a') as saved_file:
#  np.savetxt(saved_file, x, delimiter=",")
#xtr_Stat = pd.DataFrame(x)
#print(x)
#print(xtr_Stat)

#xtr_Stat=pd.read_csv('xtr_Stat_Features2.csv').iloc[:, 0:140]
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
xtr_Stat=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/xtr_Stat_Features_Xtrain_34.csv').iloc[1:46540, :]
#scaler = preprocessing.StandardScaler()
#xtr_Stat = scaler.fit_transform(xtr_Stat)


# Highly correlated features are removed
#corr_features = tsfel.correlated_features(xtr_Stat)
#xtr_Stat.drop(corr_features, axis=1, inplace=True)

#with open('corr_features.csv','a') as saved_file:
#  np.savetxt(saved_file, xtr_Stat, delimiter=",")

#Remove low variance features

#selector = VarianceThreshold(threshold=(0.9))
#xtr_Stat = selector.fit_transform(xtr_Stat)
# Normalising Features
#scaler = preprocessing.StandardScaler()
#xtr_Stat = scaler.fit_transform(xtr_Stat)

#with open('corr_features.csv','a') as saved_file:
#  np.savetxt(saved_file, xtr_Stat, delimiter=",")


#xtr=X_train.iloc[:, 0:48]
xtr_Default=X_train.iloc[:, 0:48]
xx= tsfel.feature_extraction.features.abs_energy(X_train.iloc[0, 0:48])
print(xx)
#xtr_Stat = pd.DataFrame(xx_X_train)


xtr_R=X_train.iloc[:, ::-1]
xtr1=X_train.iloc[0:23270, 0:48]
xtr2=X_train.iloc[23270:46540, 0:48]

xtr1_R=X_train.iloc[0:23270, ::-1]
xtr2_R=X_train.iloc[23270:46540, ::-1]

ytr = to_categorical(Y_train, 2)
#ytr_Stat = to_categorical(Y_train[1:20000], 2)

ytr_Stat = to_categorical(Y_train[1:46540], 2)
ytr1 = to_categorical(Y_train[0:23270], 2)
ytr2 = to_categorical(Y_train[23270:46540], 2)
xtr_h=X_train_h.iloc[:, 0:48]
#xtst=X_test.iloc[:, 0:48]
xtst_Default=X_test.iloc[:, 0:48]

#xxtst=[]
#for i in range(23139):
  #xxtst.append(smt.stattools.acovf(X_test.iloc[i, 0:48])) #autocovariance upto 20 values
#  x=smt.stattools.acf(X_test.iloc[i, 0:48],nlags=23)
#  if(np.isnan(x).any()):
#    x = np.zeros(48)
#  xxtst.append(x) 
#xtst_Stat = pd.DataFrame(xxtst)

#cfg = tsfel.get_features_by_domain()
#data_Test = []
#for i in range(5000):
#  print(i)
#  x_before = X_test.iloc[i, 0:48]
#  x_after = tsfel.time_series_features_extractor(cfg, x_before)
#  data_Test.append(x_after)

#xtst_Stat= pd.DataFrame(np.array(data_Test).reshape(10,159))

#x = np.array(data_Test).reshape(1000,159)
#with open('data_Test.csv','a') as saved_file:
#  np.savetxt(saved_file, x, delimiter=",")
xtst_Stat=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/xtr_Stat_Features_Xtest_34.csv').iloc[1:23140, :] 
 
#scaler = preprocessing.StandardScaler()
#xtst_Stat = scaler.fit_transform(xtst_Stat)
#corr_features = tsfel.correlated_features(xtst_Stat)
#xtst_Stat.drop(corr_features, axis=1, inplace=True)

# Remove low variance features
#selector = VarianceThreshold(threshold=(0.66))
#xtst_Stat = selector.fit_transform(xtst_Stat)
# Normalising Features
#scaler = preprocessing.StandardScaler()
#xtst_Stat = scaler.fit_transform(xtst_Stat)

xtst_R=X_test.iloc[:, ::-1]

ytst = to_categorical(Y_test, 2)
#ytst_Stat = to_categorical(Y_test[1:1000], 2)
ytst_Stat = to_categorical(Y_test[1:23140], 2)
xtr_1=X_train.iloc[:, 0:12]
xtr_2=X_train.iloc[:, 12:24]
xtr_3=X_train.iloc[:, 24:36]
xtr_4=X_train.iloc[:, 36:48]

xtst_1=X_test.iloc[:, 0:12]
xtst_2=X_test.iloc[:, 12:24]
xtst_3=X_test.iloc[:, 24:36]
xtst_4=X_test.iloc[:, 36:48]


##
xtr_attacker = X_train_attacker.iloc[:, 0:48];
xtr_attacker1=X_train_attacker.iloc[0:23270, 0:48]
ytr_attacker = to_categorical(Y_train_attacker, 2)
ytr_attacker1 = to_categorical(Y_train_attacker[0:23270], 2)

xtst_attacker=X_test_attacker.iloc[:, 0:48]
ytst_attacker = to_categorical(Y_test_attacker, 2)

xtst_attacker_evasion=X_test_attacker_forEvasion.iloc[:, 0:48]
xtst_attacker_evasion_R=X_test_attacker_forEvasion.iloc[:,  ::-1]
ytst_attacker_evasion = to_categorical(Y_test_attacker_forEvasion, 2)

xtst_attacker_evasion_1=X_test_attacker_forEvasion.iloc[:, 0:12]
xtst_attacker_evasion_2=X_test_attacker_forEvasion.iloc[:, 12:24]
xtst_attacker_evasion_3=X_test_attacker_forEvasion.iloc[:, 24:36]
xtst_attacker_evasion_4=X_test_attacker_forEvasion.iloc[:, 36:48]


def create_FFN_model_attacker2(input_size,l_rate): #48,0.001
  # Step 2: Create the keras model
  model = Sequential()
  model.add(Dense(100, input_dim=input_size, kernel_regularizer=regularizers.l2(l_rate)))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(700, activation='relu'))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(200, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(2, activation = 'softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
def create_FFN_model_attacker(input_size,l_rate): #48,0.001
  # Step 2: Create the keras model
  model = Sequential()
  model.add(Dense(50, input_dim=input_size, kernel_regularizer=regularizers.l2(l_rate)))
  model.add(Dense(265, activation='relu'))
  model.add(Dense(500, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(200, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(2, activation = 'softmax'))
  opt = Adam(learning_rate=l_rate)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

def create_FFN_model(m_paramters): #48,0.001
  # Step 2: Create the keras model
  #m_input_size,m_learning_rate,m_layers, m_activeFun,m_optimizer,m_metrics,m_loss
  model = Sequential()
  model.add(Dense(m_paramters._layers[0]* 100, input_dim=m_paramters._input_size, kernel_regularizer=regularizers.l2(m_paramters._learning_rate)))
  for i in range(len(m_paramters._layers)-1):
    model.add(Dense(m_paramters._layers[i+1], activation=m_paramters._activeFun))
  model.add(Dense(2, activation = 'softmax'))
  model.compile(loss=m_paramters._loss, optimizer=m_paramters._optimizer, metrics=m_paramters._metrics)
  return model


def create_GRU_model(m_paramters): #48,0.00001
  model = Sequential()
  model.add(Conv1D(m_paramters._filter1, m_paramters._filter2, activation=m_paramters._filter3, kernel_regularizer=regularizers.l2(m_paramters._learning_rate), input_shape=(m_paramters._input_size,1)))
  model.add(MaxPooling1D(pool_size=3))
  for i in range(len(m_paramters._layers)-1):
    model.add(GRU(m_paramters._layers[i], activation=m_paramters._activeFun, return_sequences=True))
  model.add(GRU(m_paramters._layers[len(m_paramters._layers)-1], activation=m_paramters._activeFun))
  model.add(Dense(2, activation = 'softmax'))
  model.compile(loss=m_paramters._loss, optimizer=m_paramters._optimizer, metrics=m_paramters._metrics)
  return model
  
def create_RNN_model(m_paramters): #48,0.00001
  model = Sequential()
  #model.add(SimpleRNN(128))  
  #model.add(GRU(128))  
  #model.add(LSTM(128)) 
  if(m_paramters._filter1 == 1):
    model.add(LSTM(units=m_paramters._filter2,dropout=m_paramters._dropout, kernel_regularizer=regularizers.l2(m_paramters._learning_rate),input_shape=(m_paramters._input_size,1)))
  elif(m_paramters._filter1 == 2):
    model.add(GRU(units=m_paramters._filter2,dropout=m_paramters._dropout, kernel_regularizer=regularizers.l2(m_paramters._learning_rate),input_shape=(m_paramters._input_size,1)))
  else:
    model.add(SimpleRNN(units=m_paramters._filter2,dropout=m_paramters._dropout, kernel_regularizer=regularizers.l2(m_paramters._learning_rate),input_shape=(m_paramters._input_size,1)))
  for i in range(len(m_paramters._layers)):
    model.add(Dropout(m_paramters._dropout))
    model.add(Dense(m_paramters._layers[i], activation=m_paramters._activeFun))
  model.add(Dense(2, activation = 'softmax'))
  model.compile(loss=m_paramters._loss, optimizer=m_paramters._optimizer, metrics=m_paramters._metrics)
  return model

def create_CNN_model(m_paramters): #48,0.00001
  model = Sequential()
  model.add(Conv1D(m_paramters._filter1, m_paramters._filter2, activation=m_paramters._activeFun, kernel_regularizer=regularizers.l2(m_paramters._learning_rate), input_shape=(m_paramters._input_size,1)))
  model.add(Dropout(m_paramters._dropout))
  for i in range(len(m_paramters._layers)):
      model.add(Dense(m_paramters._layers[i], activation=m_paramters._activeFun))
      model.add(Dropout(m_paramters._dropout))
  model.add(MaxPooling1D())
  model.add(Flatten())
  model.add(Dense(2, activation = 'softmax'))
  #opt = Adam(learning_rate=l_rate)
  model.compile(loss=m_paramters._loss, optimizer=m_paramters._optimizer, metrics=m_paramters._metrics)
  return model

class M_Paramters:
  def __init__(self, _filter1, _filter2,_filter3,_dropout,_activeFun,_input_size,_learning_rate,_layers,_loss,_optimizer,_metrics):
    self._filter1 = _filter1
    self._filter2 = _filter2
    self._filter3 =_filter3
    self._dropout = _dropout
    self._activeFun = _activeFun
    self._input_size = _input_size
    self._learning_rate = _learning_rate
    self._layers = _layers
    self._loss = _loss
    self._optimizer = _optimizer
    self._metrics = _metrics

#####################################################
############# Attacker FNN ###################################
#####################################################

#model_attacker = create_FFN_model_attacker(48,0.00001);
#classifier_attacker = KerasClassifier(model=model_attacker,clip_values=(0, 10000000))

#classifier_attacker.fit(xtr_attacker, ytr_attacker, batch_size=128, nb_epochs=10, verbose=1)
#model_attacker.save("xdata/modelFNN_attacker.h5")
model_attacker = tf.keras.models.load_model("xdata/modelFNN_attacker.h5")
classifier_attacker = KerasClassifier(model=model_attacker,clip_values=(0, 10000000))

#####################################################
############# FNN ###################################
#####################################################

#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']


arr_m_paramters_FNN = []
arr_models_FNN = []
arr_classifier_FNN = []
num_models_FNN = 7
paramter_FNN_1 = M_Paramters(0, 0,0,None,'relu',48,0.00001,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])
paramter_FNN_2 = M_Paramters(0, 0,0,None,'relu',34,0.00001,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])
paramter_FNN_3 = M_Paramters(0, 0,0,None,'relu',34,0.00001,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])
paramter_FNN_4 = M_Paramters(0, 0,0,None,'relu',34,0.02,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.02),['accuracy'])
paramter_FNN_5 = M_Paramters(0, 0,0,None,'relu',34,0.00001,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])
paramter_FNN_6 = M_Paramters(0, 0,0,None,'relu',34,0.001,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.001),['accuracy'])
paramter_FNN_7 = M_Paramters(0, 0,0,None,'relu',34,0.0002,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.0002),['accuracy'])
paramter_FNN_8 = M_Paramters(0, 0,0,None,'relu',34,0.0002,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.0002),['accuracy'])
paramter_FNN_9 = M_Paramters(0, 0,0,None,'relu',34,0.00001,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])
paramter_FNN_10 = M_Paramters(0, 0,0,None,'relu',5,0.00001,[100,512,700,512,256,840,1024,768,256,200,50],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])

paramter_FNN_33 = M_Paramters(0, 0,0,None,'relu',143,0.00001,[100,512,700,200,50],'sparse_categorical_crossentropy',Adagrad(learning_rate=0.0001),['accuracy'])
paramter_FNN_44 = M_Paramters(0, 0,0,None,'relu',143,0.02,[100,512,700,512,256,840,1024,768,256,200,50],'sparse_categorical_crossentropy',SGD(learning_rate=0.02),['accuracy'])
paramter_FNN_55 = M_Paramters(0, 0,0,None,'sigmoid',143,0.003,[100,512,700,200,50],'categorical_crossentropy',Adagrad(learning_rate=0.003),['accuracy'])
paramter_FNN_66 = M_Paramters(0, 0,0,None,'tanh',143,0.1,[100,512,700,200,50],'categorical_crossentropy',Adagrad(learning_rate=0.1),['accuracy'])
paramter_FNN_77 = M_Paramters(0, 0,0,None,'hard_sigmoid',143,0.001,[100,512,700,200,50],'categorical_crossentropy',Adagrad(learning_rate=0.001),['accuracy'])
paramter_FNN_88 = M_Paramters(0, 0,0,None,'relu',143,0.001,[100,512,700,200,50],'categorical_crossentropy',Nadam(learning_rate=0.001),['accuracy'])
paramter_FNN_99 = M_Paramters(0, 0,0,None,'relu',143,0.00001,[100,512,700,512,200,50],'sparse_categorical_crossentropy',SGD(learning_rate=0.00001),['accuracy'])
paramter_FNN_1010 = M_Paramters(0, 0,0,None,'softsign',143,0.00001,[100,512,700,512,200,50],'categorical_crossentropy',Adadelta(learning_rate=0.00001),['accuracy'])

arr_m_paramters_FNN.append(paramter_FNN_1)
arr_m_paramters_FNN.append(paramter_FNN_2)
arr_m_paramters_FNN.append(paramter_FNN_3)
arr_m_paramters_FNN.append(paramter_FNN_4)
arr_m_paramters_FNN.append(paramter_FNN_5)
arr_m_paramters_FNN.append(paramter_FNN_6)
arr_m_paramters_FNN.append(paramter_FNN_7)
arr_m_paramters_FNN.append(paramter_FNN_8)
arr_m_paramters_FNN.append(paramter_FNN_9)
arr_m_paramters_FNN.append(paramter_FNN_10)


for i in range(num_models_FNN):
    m = create_FFN_model(arr_m_paramters_FNN[i])
    arr_models_FNN.append(m)

for i in range(num_models_FNN):
    c = KerasClassifier(model=arr_models_FNN[i])
    arr_classifier_FNN.append(c)        



for i in range(num_models_FNN): 
    if(i ==0):
        arr_classifier_FNN[i].fit(xtr_Default , ytr , batch_size= 128, nb_epochs= (1), verbose=1)
    elif( i==1):
        arr_classifier_FNN[i].fit(xtr_Stat, ytr_Stat, batch_size=128, nb_epochs= (1), verbose=1)
    elif( i==2):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    elif( i==3):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    elif( i==4):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    elif( i==5):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    elif( i==6):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    elif( i==7):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    elif( i==8):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    elif( i==9):
        arr_classifier_FNN[i].fit(xtr_Stat.iloc[:,0:34], ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    arr_models_FNN[i].save("xdata/modelFNN_"+ str(i+1) +".h5")

#arr_classifier_FNN[i].fit(xtr_Stat , ytr_Stat , batch_size= 128, nb_epochs= (1), verbose=1)
   
# fit all models    
n_members = 5
# evaluate averaging ensemble (equal weights)
weights = [1.0/n_members for _ in range(n_members)]
members_all = [arr_classifier_FNN[2],arr_classifier_FNN[3],arr_classifier_FNN[4],arr_classifier_FNN[5],arr_classifier_FNN[6]]
#####################################################
############# RNN ###################################
#####################################################
arr_m_paramters_RNN = []
arr_models_RNN = []
arr_classifier_RNN = []
num_models_RNN = 0
paramter_RNN_1 = M_Paramters(1, 55,None,0.05,'relu',55,0.0002,[265,128,64,32],'categorical_crossentropy',Adam(learning_rate=0.0002),['accuracy'])
paramter_RNN_2 = M_Paramters(1, 55,None,0.05,'relu',55,0.0002,[265,128,64,32],'categorical_crossentropy',Adam(learning_rate=0.0002),['accuracy'])
paramter_RNN_3 = M_Paramters(3, 48,None,0.05,'relu',48,0.003,[265,128,64,32],'categorical_crossentropy',Adagrad(learning_rate=0.003),['accuracy'])
paramter_RNN_4 = M_Paramters(2, 48,None,0.05,'relu',48,0.002,[265,128,64,32],'categorical_crossentropy',Adam(learning_rate=0.002),['accuracy'])
paramter_RNN_5 = M_Paramters(3, 48,None,0.05,'relu',48,0.003,[265,128,64,32],'categorical_crossentropy',Adagrad(learning_rate=0.003),['accuracy'])

arr_m_paramters_RNN.append(paramter_RNN_1)
arr_m_paramters_RNN.append(paramter_RNN_2)
arr_m_paramters_RNN.append(paramter_RNN_3)
arr_m_paramters_RNN.append(paramter_RNN_4)
arr_m_paramters_RNN.append(paramter_RNN_5)

for i in range(num_models_RNN):
    m = create_RNN_model(arr_m_paramters_RNN[i])
    arr_models_RNN.append(m)

for i in range(num_models_RNN):
    c = KerasClassifier(model=arr_models_RNN[i])
    arr_classifier_RNN.append(c)        

for i in range(num_models_RNN):
    arr_classifier_RNN[i].fit(xtr_Stat.reshape((19999, 55,1)), ytr_Stat, batch_size=128, nb_epochs= (10), verbose=1)
    #if(i==0):
    #    arr_classifier_RNN[i].fit(xtr_Stat.values.reshape((46539, 48,1)), ytr, batch_size=128, nb_epochs= (10), verbose=1)
    #else:
    #    arr_classifier_RNN[i].fit(xtr_Stat.values.reshape((46539, 48,1)), ytr, batch_size=128, nb_epochs= (10), verbose=1)
    #arr_models_RNN[i].save("xdata/modelRNN_"+ str(i+1) +".h5")


#####################################################
############# GRU ###################################
#####################################################
arr_m_paramters_GRU = []
arr_models_GRU = []
arr_classifier_GRU = []
num_models_GRU = 0
paramter_GRU_1 = M_Paramters(64, 3,'relu',None,'tanh',32,0.002,[64,64,64,64],'categorical_crossentropy',Adam(learning_rate=0.002),['accuracy'])
paramter_GRU_2 = M_Paramters(64, 3,'relu',None,'tanh',32,0.002,[64,64,64,64],'categorical_crossentropy',Adam(learning_rate=0.002),['accuracy'])
paramter_GRU_3 = M_Paramters(64, 3,'relu',None,'tanh',48,0.0001,[64,64,64,64],'sparse_categorical_crossentropy',Adagrad(learning_rate=0.0001),['accuracy'])
paramter_GRU_4 = M_Paramters(128, 6,'relu',None,'tanh',48,0.0002,[64,64,64,64],'sparse_categorical_crossentropy',SGD(learning_rate=0.0002),['accuracy'])
paramter_GRU_5 = M_Paramters(128, 6,'relu',None,'tanh',48,0.003,[64,64,64,64],'categorical_crossentropy',Adagrad(learning_rate=0.003),['accuracy'])

arr_m_paramters_GRU.append(paramter_GRU_1)
arr_m_paramters_GRU.append(paramter_GRU_2)
arr_m_paramters_GRU.append(paramter_GRU_3)
arr_m_paramters_GRU.append(paramter_GRU_4)
arr_m_paramters_GRU.append(paramter_GRU_5)

for i in range(num_models_GRU):
    m = create_GRU_model(arr_m_paramters_GRU[i])
    arr_models_GRU.append(m)

for i in range(num_models_GRU):
    c = KerasClassifier(model=arr_models_GRU[i])
    arr_classifier_GRU.append(c)        

for i in range(num_models_GRU):
    arr_classifier_GRU[i].fit(xtr_Stat.reshape((19999, 32,1)), ytr_Stat, batch_size=128, nb_epochs= (2), verbose=1)
    #if(i==0):
    #    arr_classifier_GRU[i].fit(xtr_Stat.values.reshape((46539, 48,1)), ytr, batch_size=128, nb_epochs= (2), verbose=1)
    #else:
    #    arr_classifier_GRU[i].fit(xtr_Stat.values.reshape((46539, 48,1)), ytr, batch_size=128, nb_epochs= (2), verbose=1)
    #arr_models_GRU[i].save("xdata/modelGRU_"+ str(i+1) +".h5")



#####################################################
############# CNN ###################################
#####################################################
arr_m_paramters_CNN = []
arr_models_CNN = []
arr_classifier_CNN = []
num_models_CNN = 0
paramter_CNN_1 = M_Paramters(128, 6,None,0.05,'relu',48,0.00001,[64,32],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])
paramter_CNN_2 = M_Paramters(128, 6,None,0.05,'relu',48,0.00001,[64,32],'categorical_crossentropy',Adam(learning_rate=0.00001),['accuracy'])
paramter_CNN_3 = M_Paramters(64, 3,None,0.05,'relu',48,0.00001,[64,32],'sparse_categorical_crossentropy',Adagrad(learning_rate=0.00001),['accuracy'])
paramter_CNN_4 = M_Paramters(128, 6,None,0.05,'relu',48,0.0002,[128,64,32],'sparse_categorical_crossentropy',SGD(learning_rate=0.0002),['accuracy'])
paramter_CNN_5 = M_Paramters(256, 9,None,0.05,'relu',48,0.003,[128,64,32],'categorical_crossentropy',Adagrad(learning_rate=0.003),['accuracy'])

arr_m_paramters_CNN.append(paramter_CNN_1)
arr_m_paramters_CNN.append(paramter_CNN_2)
arr_m_paramters_CNN.append(paramter_CNN_3)
arr_m_paramters_CNN.append(paramter_CNN_4)
arr_m_paramters_CNN.append(paramter_CNN_5)


for i in range(num_models_CNN):
    m = create_CNN_model(arr_m_paramters_CNN[i])
    arr_models_CNN.append(m)

for i in range(num_models_CNN):
    c = KerasClassifier(model=arr_models_CNN[i])
    arr_classifier_CNN.append(c)        

for i in range(num_models_CNN):
    if(i== 0):
        arr_classifier_CNN[i].fit(xtr_Stat.values.reshape((46539, 48,1)), ytr, batch_size=128, nb_epochs= (10), verbose=1)
    else:
        arr_classifier_CNN[i].fit(xtr_Stat.values.reshape((46539, 48,1)), ytr, batch_size=128, nb_epochs= (10), verbose=1)
    arr_models_CNN[i].save("xdata/modelRNN_"+ str(i+1) +".h5")



#####################################################
############# Test bengin&attack samples ###################################
#####################################################

predictions = classifier_attacker.predict(xtst_attacker)
display_metric(predictions,ytst,"attacker","test bengin&attack samples")


for i in range(num_models_FNN):
  if(i ==0):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Default)
    display_metric(predictions1,ytst,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==1):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat)
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==2):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==3):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==4):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==5):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==6):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==7):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==8):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
  elif(i==9):
    predictions1 = arr_classifier_FNN[i].predict(xtst_Stat.iloc[:,0:34])
    display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")
#predictions1 = arr_classifier_FNN[i].predict(xtst_Stat)
#display_metric(predictions1,ytst_Stat,("FFN_all_"+str(i+1)),"test bengin&attack samples")


for i in range(num_models_RNN):
  if(i ==0):
    predictions1 = arr_classifier_RNN[i].predict(xtst_Stat.values.reshape((23139, 48,1)))
  else:
    predictions1 = arr_classifier_RNN[i].predict(xtst_Stat.values.reshape((23139, 48,1)))
  display_metric(predictions1,ytst,("RNN_all_"+str(i+1)),"test bengin&attack samples")

for i in range(num_models_GRU):
  if(i ==0):
    predictions1 = arr_classifier_GRU[i].predict(xtst_Stat.values.reshape((23139, 48,1)))
  else:
    predictions1 = arr_classifier_GRU[i].predict(xtst_Stat.values.reshape((23139, 48,1)))
  display_metric(predictions1,ytst,("GRU_all_"+str(i+1)),"test bengin&attack samples")



for i in range(num_models_CNN):
    if(i ==0):
      predictions1 = arr_classifier_CNN[i].predict(xtst_Stat.values.reshape((23139, 48,1)))
    else:
      predictions1 = arr_classifier_CNN[i].predict(xtst_Stat.values.reshape((23139, 48,1)))
    display_metric(predictions1,ytst,("CNN_all_"+str(i+1)),"test bengin&attack samples")


#####################################################
############# Generate Adv samples ###################################
#####################################################

attack = FastGradientMethod(estimator=classifier_attacker, eps=0.2)
x_test_adv = attack.generate(x=xtst_attacker_evasion.values)


#with open('xEvaion_Stat_Attack.csv','a') as saved_file:
#  np.savetxt(saved_file, xtst_attacker_evasion, delimiter=",")

#with open('xEvaion_Stat_Adver.csv','a') as saved_file:
#  np.savetxt(saved_file, x_test_adv, delimiter=",")

#data_Evasion = []
#for i in range(1000):
#  print(i)
#  x_before = xtst_attacker_evasion.values[i, 0:48]
#  x_after = tsfel.time_series_features_extractor(cfg, x_before)
#  data_Evasion.append(x_after)

#xEvaion_Stat2= pd.DataFrame(np.array(data_Evasion).reshape(1000,159))

#with open('xEvaion_Stat_Attack.csv','a') as saved_file:
#  np.savetxt(saved_file, xEvaion_Stat2, delimiter=",")

#cfg = tsfel.get_features_by_domain()
#data_Evasion = []
#for i in range(1000):
#  print(i)
#  x_before = x_test_adv[i, 0:48]
#  x_after = tsfel.time_series_features_extractor(cfg, x_before)
#  data_Evasion.append(x_after)

#xEvaion_Stat= pd.DataFrame(np.array(data_Evasion).reshape(1000,159))

#with open('xEvaion_Stat_Adver.csv','a') as saved_file:
#  np.savetxt(saved_file, xEvaion_Stat, delimiter=",")

#corr_features = tsfel.correlated_features(xtst_Stat)
#xtst_Stat.drop(corr_features, axis=1, inplace=True)

# Remove low variance features
#selector = VarianceThreshold(threshold=(0.004))
#xEvaion_Stat = selector.fit_transform(xEvaion_Stat)
# Normalising Features
#scaler = preprocessing.StandardScaler()
#xEvaion_Stat = scaler.fit_transform(xEvaion_Stat)


#corr_features = tsfel.correlated_features(xtst_Stat)
#xtst_Stat.drop(corr_features, axis=1, inplace=True)

# Remove low variance features
#selector = VarianceThreshold(threshold=(0.004))
#xEvaion_Stat2 = selector.fit_transform(xEvaion_Stat2)
# Normalising Features
#scaler = preprocessing.StandardScaler()
#xEvaion_Stat2 = scaler.fit_transform(xEvaion_Stat2)


#acovf
xtst_attacker_evasion2=[]
for i in range(11569):
  #xtst_attacker_evasion2.append(smt.stattools.acovf(xtst_attacker_evasion.values[i, 0:48])) #autocovariance upto 20 values
  x =smt.stattools.acf(xtst_attacker_evasion.values[i, 0:48],nlags=23)
  if(np.isnan(x).any()):
    x = np.zeros(48)
  xtst_attacker_evasion2.append(x) 
xtst_attacker_evasion2 = pd.DataFrame(xtst_attacker_evasion2)
x_test_adv2=[]
for i in range(11569):
  #x_test_adv2.append(smt.stattools.acovf(x_test_adv[i, 0:48])) #autocovariance upto 20 values
  x =smt.stattools.acf(x_test_adv[i, 0:48],nlags=23)
  if(np.isnan(x).any()):
    x = np.zeros(48)
  x_test_adv2.append(x)
x_test_adv2 = np.array(x_test_adv2)


xEvaion_Stat2=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/x_attacker_all_34.csv').iloc[1:11569:, :]  
#scaler = preprocessing.StandardScaler()
#xEvaion_Stat2 = scaler.fit_transform(xEvaion_Stat2)
xEvaion_Stat=pd.read_csv('drive/MyDrive/myColabData/Data/65Defender/x_advar_all_34.csv').iloc[1:11569:, :]  
#scaler = preprocessing.StandardScaler()
#xEvaion_Stat = scaler.fit_transform(xEvaion_Stat)


#####################################################
############# Test Attacks & Adv samples ###################################
#####################################################

predictions_before_attacker = classifier_attacker.predict(xtst_attacker_evasion)
display_Accuracy(predictions_before_attacker,ytst_attacker_evasion,"FFN _attacker","attacks samples")
predictions_after_attacker = classifier_attacker.predict(x_test_adv)
display_Accuracy(predictions_after_attacker,ytst_attacker_evasion,"FFN _attacker","adversarial samples")


for i in range(num_models_FNN):
   if( i==0):
     predictions_before1 = arr_classifier_FNN[i].predict(xtst_attacker_evasion)
     display_Accuracy(predictions_before1,ytst_attacker_evasion,("FFN_all_"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(x_test_adv)
     display_Accuracy(predictions_after1,ytst_attacker_evasion,("FFN_all_"+str(i+1)),"adversarial samples")
   elif(i==1):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2)
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat)
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==2):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==3):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==4):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==5):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==6):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==7):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==8):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")
   elif(i==9):
     predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2.iloc[:,0:34])
     display_Accuracy(predictions_before1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"attacks samples")
     predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat.iloc[:,0:34])
     display_Accuracy(predictions_after1,ytst_attacker_evasion[0:11567],("FFN_all__"+str(i+1)),"adversarial samples")

#predictions_before1 = arr_classifier_FNN[i].predict(xEvaion_Stat2)
#display_Accuracy(predictions_before1,ytst_attacker_evasion[0:999],("FFN_all_"+str(i+1)),"attacks samples")
#predictions_after1 = arr_classifier_FNN[i].predict(xEvaion_Stat)
#display_Accuracy(predictions_after1,ytst_attacker_evasion[0:999],("FFN_all__"+str(i+1)),"adversarial samples")
score = evaluate_ensemble(members_all, weights,xEvaion_Stat2.iloc[:,0:34], xEvaion_Stat2.iloc[:,0:34], xEvaion_Stat2.iloc[:,0:34], xEvaion_Stat2.iloc[:,0:34], xEvaion_Stat2.iloc[:,0:34],Y_test_attacker_forEvasion.iloc[0:11567])
print('Ensemble Equal Weights Score attack samples : %.3f' % score)
score = evaluate_ensemble(members_all, weights,xEvaion_Stat.iloc[:,0:34], xEvaion_Stat.iloc[:,0:34], xEvaion_Stat.iloc[:,0:34], xEvaion_Stat.iloc[:,0:34], xEvaion_Stat.iloc[:,0:34],Y_test_attacker_forEvasion.iloc[0:11567])
print('Ensemble Equal Weights Score adversarial samples : %.3f' % score)

for i in range(num_models_RNN):
  if(i == 0):
    predictions_before1 = arr_classifier_RNN[i].predict(xtst_attacker_evasion2.values.reshape((11569, 48,1)))
    display_Accuracy(predictions_before1,ytst_attacker_evasion,("RNN_all_"+str(i+1)),"attacks samples")
    predictions_after1 = arr_classifier_RNN[i].predict(x_test_adv2.reshape((11569, 48,1)))
    display_Accuracy(predictions_after1,ytst_attacker_evasion,("RNN_all_"+str(i+1)),"adversarial samples")
  else:
    predictions_before1 = arr_classifier_RNN[i].predict(xtst_attacker_evasion2.values.reshape((11569, 48,1)))
    display_Accuracy(predictions_before1,ytst_attacker_evasion,("RNN_all_"+str(i+1)),"attacks samples")
    predictions_after1 = arr_classifier_RNN[i].predict(x_test_adv2.reshape((11569, 48,1)))
    display_Accuracy(predictions_after1,ytst_attacker_evasion,("RNN_all_"+str(i+1)),"adversarial samples")

for i in range(num_models_GRU):
  if(i == 0):
    predictions_before1 = arr_classifier_GRU[i].predict(xtst_attacker_evasion2.values.reshape((11569, 48,1)))
    display_Accuracy(predictions_before1,ytst_attacker_evasion,("GRU_all_"+str(i+1)),"attacks samples")
    predictions_after1 = arr_classifier_GRU[i].predict(x_test_adv2.reshape((11569, 48,1)))
    display_Accuracy(predictions_after1,ytst_attacker_evasion,("GRU_all_"+str(i+1)),"adversarial samples")
  else:
    predictions_before1 = arr_classifier_GRU[i].predict(xtst_attacker_evasion2.values.reshape((11569, 48,1)))
    display_Accuracy(predictions_before1,ytst_attacker_evasion,("GRU_all_"+str(i+1)),"attacks samples")
    predictions_after1 = arr_classifier_GRU[i].predict(x_test_adv2.reshape((11569, 48,1)))
    display_Accuracy(predictions_after1,ytst_attacker_evasion,("GRU_all_"+str(i+1)),"adversarial samples")

for i in range(num_models_CNN):
  if(i == 0):
    predictions_before1 = arr_classifier_CNN[i].predict(xtst_attacker_evasion2.values.reshape((11569, 48,1)))
    display_Accuracy(predictions_before1,ytst_attacker_evasion,("CNN_all_"+str(i+1)),"attacks samples")
    predictions_after1 = arr_classifier_CNN[i].predict(x_test_adv2.reshape((11569, 48,1)))
    display_Accuracy(predictions_after1,ytst_attacker_evasion,("CNN_all_"+str(i+1)),"adversarial samples")
  else:
    predictions_before1 = arr_classifier_CNN[i].predict(xtst_attacker_evasion2.values.reshape((11569, 48,1)))
    display_Accuracy(predictions_before1,ytst_attacker_evasion,("CNN_all_"+str(i+1)),"attacks samples")
    predictions_after1 = arr_classifier_CNN[i].predict(x_test_adv2.reshape((11569, 48,1)))
    display_Accuracy(predictions_after1,ytst_attacker_evasion,("CNN_all_"+str(i+1)),"adversarial samples")

#####################################################
#####################################################
