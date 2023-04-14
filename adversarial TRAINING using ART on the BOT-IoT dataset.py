# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:19:51 2020

@author: Adam
"""
#%%

import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import BasicIterativeMethod,ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%%
filename = "UNSW_2018_IoT_Botnet_Final_10_Best.csv"
df = pd.read_csv(filename,low_memory=False)

df=df.drop(['index','pkSeqID','category', 'subcategory'], axis='columns')
df_400=df.groupby('attack').apply(lambda x: x.sample(400))
df_400.head()

#pip install category_encoders
import category_encoders as ce

# create an object of the OrdinalEncoding
ce_ordinal = ce.OrdinalEncoder(cols=['proto','saddr','sport','daddr','dport'])
# fit and transform and you will get the encoded data
data_OrdinalEncoder=ce_ordinal.fit_transform(df_400)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

n_features = 15
n_classes = 2

X = data_OrdinalEncoder[[ 'proto', 'saddr', 'sport','daddr','dport','seq','stddev','N_IN_Conn_P_SrcIP', 'min', 'state_number', 'mean','N_IN_Conn_P_DstIP', 'drate', 'srate', 'max']]
y = data_OrdinalEncoder.attack.to_numpy()

# One hot encoding
yHEC= tf.keras.utils.to_categorical(y, n_classes)

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# =============================================================================
# #saving a normalized version of the data as a csv file
# data_scaled = scaler.fit_transform(data)
# np.savetxt("data_scaled.csv", data_scaled, delimiter=",")
# =============================================================================

# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, yHEC, test_size=0.2, random_state=2)

#%%
# =============================================================================
# (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
# =============================================================================

#%%

def accuracyPerso(classifier, X_test_, Y_test):
    
    Y_pred=np.argmax(classifier.predict(X_test_), axis=1)
    label = np.argmax(Y_test, axis = 1)
    nb_correct_pred = np.sum(Y_pred == label)
    print('\nAccuracy of the model on test data: {:4.2f}%'.format(nb_correct_pred/label.shape[0] * 100))
    
    from sklearn.metrics import precision_recall_fscore_support as score
    from sklearn.metrics import confusion_matrix
    
    precision, recall, fscore, support = score(label, Y_pred)
    CM=confusion_matrix(label, Y_pred)
    
    print('confusion_matrix:\n {}'.format(CM))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return nb_correct_pred, nb_correct_pred/label.shape[0]
#%%
model = tf.keras.models.Sequential(
    [   
        tf.keras.layers.Dense(n_features, activation=tf.nn.relu,input_dim = n_features),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
    ]
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



#%%
# =============================================================================
# path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH,
#                 url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1')
# classifier_model = load_model(path)
# =============================================================================
model.fit(X_train, Y_train, epochs=50,verbose=1)
classifier = KerasClassifier( model=model)


#%%
x_test_pred = np.argmax(classifier.predict(X_test), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(Y_test, axis=1))

print("Original test data :")
print("Correctly classified: {}".format(nb_correct_pred))
print("Incorrectly classified: {}".format(Y_test.shape[0]-nb_correct_pred))
#%%

attacker = FastGradientMethod(classifier, eps=2)
x_test_adv = attacker.generate(X_test) 
#%%
x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(Y_test, axis=1))

print("Adversarial test data :")
print("Correctly classified: {}".format(nb_correct_adv_pred))
print("Incorrectly classified: {}".format(Y_test.shape[0]-nb_correct_adv_pred))

#%%
robust_classifier_model = tf.keras.models.Sequential(
    [   
        tf.keras.layers.Dense(n_features, activation=tf.nn.relu,input_dim = n_features),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
    ]
)

robust_classifier_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

robust_classifier_model.summary()



#%%

# =============================================================================
# path = get_file('mnist_cnn_robust.h5', extract=False, path=ART_DATA_PATH,
#                 url='https://www.dropbox.com/s/yutsncaniiy5uy8/mnist_cnn_robust.h5?dl=1')
# robust_classifier_model = load_model(path)
# =============================================================================

robust_classifier = KerasClassifier(model=robust_classifier_model, use_logits=False)

#%%
Training_attack = BasicIterativeMethod(robust_classifier, eps=0.9, eps_step=0.01, max_iter=100)

# Here is the command we had used for the Adversarial Training

trainer = AdversarialTrainer(robust_classifier, Training_attack, ratio=0.99)
trainer.fit(X_train, Y_train, nb_epochs=20, batch_size=20)

x_test_robust_pred = np.argmax(robust_classifier.predict(X_test), axis=1)
nb_correct_robust_pred = np.sum(x_test_robust_pred == np.argmax(Y_test, axis=1))

print("Original test data :")
print("Correctly classified: {}".format(nb_correct_robust_pred))
print("Incorrectly classified: {}".format(Y_test.shape[0]-nb_correct_robust_pred))


attacker_robust = BasicIterativeMethod(robust_classifier, eps=0.9)
x_test_adv_robust = attacker_robust.generate(X_test)

x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == np.argmax(Y_test, axis=1))

print("Adversarial test data :")
print("Correctly classified: {}".format(nb_correct_adv_robust_pred))
print("Incorrectly classified: {}".format(Y_test.shape[0]-nb_correct_adv_robust_pred))
#%%
Training_attack = ProjectedGradientDescent(robust_classifier, eps=0.9)
attacker_robust = ProjectedGradientDescent(robust_classifier, eps=0.9)

eps_range = [0.1, 0.3, 0.5, 0.7, 0.9,1.1,1.3, 1.5,1.7, 2]
ratio_range = [0.3, 0.5, 0.7, 0.9]
nb_correct_original = np.zeros((len(ratio_range),len(eps_range)))
nb_correct_robust = np.zeros((len(ratio_range),len(eps_range),len(eps_range)))

for r,ratio_val in enumerate(ratio_range):
    print("\n r=",r,"***************** ratio_val=",ratio_val)
    for i,eps_def in enumerate(eps_range):
        print("\n i=",i,"++++++++++++++++ eps_def=",eps_def)
        Training_attack.set_params(**{'eps': eps_def})
        trainer = AdversarialTrainer(robust_classifier, Training_attack, ratio=ratio_val)
        trainer.fit(X_train, Y_train, nb_epochs=20, batch_size=20)
        x_test_robust_pred = np.argmax(robust_classifier.predict(X_test), axis=1)
        nb_correct_original[r,i] = np.sum(x_test_robust_pred == np.argmax(Y_test, axis=1))/Y_test.shape[0]
        
        for j,eps_att in enumerate(eps_range):
            print("\n j=",j,"------------------ eps_att=",eps_att)
            attacker_robust.set_params(**{'eps': eps_att})
            x_test_adv_robust = attacker_robust.generate(X_test)
            x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
            nb_correct_robust[r,i,j] = np.sum(x_test_adv_robust_pred == np.argmax(Y_test, axis=1))/Y_test.shape[0]
            
#%%

eps_range = [0.1, 0.3, 0.5, 0.7, 0.9,1.1,1.3, 1.5,1.7, 2.0]
attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=10)
nb_correct_attack_pgd = []


for eps in eps_range:
    print("eps=",eps)

    attack_pgd.set_params(**{'eps': eps})

    X_test_adv_pgd = attack_pgd.generate(X_test)
     
    x_test_pred = np.argmax(classifier.predict(X_test_adv_pgd), axis=1)
    nb_correct_attack_pgd += [np.sum(x_test_pred == np.argmax(Y_test, axis=1))/Y_test.shape[0]]
    
#%%
#del_nb_correct_robust = nb_correct_robust[:,:-1,:-1]
import matplotlib.pyplot as plt
for r,ratio_val in enumerate(ratio_range):
    fig, ax = plt.subplots()
    
    ax.plot(np.array(eps_range), np.array(nb_correct_original[r,:]), 'g--', label='Clean test data')
    
    for i,eps_def in enumerate(eps_range):
        label='eps_def='+ str(eps_def)
        ax.plot(np.array(eps_range), np.array(nb_correct_robust[r,i,:]), '--', label=label)
        
    ax.plot(np.array(eps_range), np.array(nb_correct_attack_pgd), 'k--', label='PGD attack \nwithout adv training')
    legend = ax.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    ax.legend(bbox_to_anchor=(1, 1))
    title_name='Ratio of adversarial training samples ='+str(ratio_val)
    plt.title(title_name)
    plt.xlabel('Attack strength (eps)')
    plt.ylabel('Accuracy')
    plt.show()
#%%    
   
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for r,ratio_val in enumerate(ratio_range):
    label='ratio='+ str(ratio_val)
    ax.plot(np.array(eps_range), np.array(nb_correct_original[r,:]), '--', label=label)



legend = ax.legend(loc='best', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#00FFCC')
ax.legend(bbox_to_anchor=(1, 1))

plt.title('effect of adversarial training on clean test classification')
plt.xlabel('Attack strength (eps)')
plt.ylabel('Accuracy')
plt.show()
    
#%%

Training_attack = ProjectedGradientDescent(robust_classifier, eps=0.9)


eps_range = [ 1 , 5, 10, 15, 20]
ratio_range = [0.1, 0.7, 0.9, 1]
nb_correct_original = np.zeros((len(ratio_range),len(eps_range)))

for r,ratio_val in enumerate(ratio_range):
    
    for i,eps_def in enumerate(eps_range):
        
        Training_attack.set_params(**{'eps': eps_def})
        trainer = AdversarialTrainer(robust_classifier, Training_attack, ratio=ratio_val)
        trainer.fit(X_train, Y_train, nb_epochs=10, batch_size=20)
        x_test_robust_pred = np.argmax(robust_classifier.predict(X_test), axis=1)
        nb_correct_original[r,i] = np.sum(x_test_robust_pred == np.argmax(Y_test, axis=1))/Y_test.shape[0]
  
#%%

# =============================================================================
#
# eps_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0, 5.0]
# ratio_range = [0.2, 0.4, 0.6, 0.8, 1]
# 
# nb_correct_original = []
# nb_correct_robust = np.zeros((len(eps_range),len(eps_range)))
# 
# for r,ratio_val in enumerate(ratio_range):
#     
#     for i,eps_def in enumerate(eps_range):
#     
#         for j,eps_att in enumerate(eps_range):
#             nb_correct_robust[r,i,j]=eps_att
# 
# 
# 
# #%%
# eps_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# nb_correct_original = []
# nb_correct_robust = []
# 
# for eps in eps_range:
#     attacker.set_params(**{'eps': eps})
#     attacker_robust.set_params(**{'eps': eps})
#     x_test_adv = attacker.generate(X_test)
#     x_test_adv_robust = attacker_robust.generate(X_test)
#     
#     x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
#     nb_correct_original += [np.sum(x_test_adv_pred == np.argmax(Y_test, axis=1))]
#     
#     x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
#     nb_correct_robust += [np.sum(x_test_adv_robust_pred == np.argmax(Y_test, axis=1))]
# 
# eps_range = [0] + eps_range
# nb_correct_original = [nb_correct_pred] + nb_correct_original
# nb_correct_robust = [nb_correct_robust_pred] + nb_correct_robust
# #%%
# fig, ax = plt.subplots()
# ax.plot(np.array(eps_range), np.array(nb_correct_original), 'b--', label='Original classifier')
# ax.plot(np.array(eps_range), np.array(nb_correct_robust), 'r--', label='Robust classifier')
# 
# legend = ax.legend(loc='lower center', shadow=True, fontsize='large')
# legend.get_frame().set_facecolor('#00FFCC')
# 
# plt.xlabel('Attack strength (eps)')
# plt.ylabel('Correct predictions')
# plt.show()
# #%%
# accuracyPerso(classifier, x_test_adv, Y_test)
# #%%
# accuracyPerso(robust_classifier, x_test_adv, Y_test)
# #%%
# =============================================================================

#%%

        
#%%

#%%

