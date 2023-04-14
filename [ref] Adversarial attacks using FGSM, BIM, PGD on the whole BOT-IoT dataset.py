# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:29:16 2020

@author: Adam
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from matplotlib import pyplot as plt

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, CarliniLInfMethod,BasicIterativeMethod, ProjectedGradientDescent


#%%
filename = "UNSW_2018_IoT_Botnet_Final_10_Best.csv"
df = pd.read_csv(filename,low_memory=False)

df=df.drop(['index','pkSeqID','category', 'subcategory'], axis='columns')
df.head()

#%%
#pip install category_encoders
import category_encoders as ce

# create an object of the OrdinalEncoding
ce_ordinal = ce.OrdinalEncoder(cols=['proto','saddr','sport','daddr','dport'])
# fit and transform and you will get the encoded data
data_OrdinalEncoder=ce_ordinal.fit_transform(df)

# =============================================================================
# #%%
# 
# # create an object of the OneHotEncoder
# ce_OHE = ce.OneHotEncoder(cols=['proto','saddr','daddr'])
# # fit and transform and you will get the encoded data
# data_OneHotEncoder=ce_OHE.fit_transform(df)
# =============================================================================

#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data_OrdinalEncoder[[ 'proto', 'saddr', 'sport','daddr','dport','seq','stddev','N_IN_Conn_P_SrcIP', 'min', 'state_number', 'mean','N_IN_Conn_P_DstIP', 'drate', 'srate', 'max']]
y = data_OrdinalEncoder.attack

# =============================================================================
# # One hot encoding
#enc = OneHotEncoder()
#Y = enc.fit_transform(y[:, np.newaxis]).toarray()
# =============================================================================

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
    X_scaled, y, test_size=0.2, random_state=2)

n_features = 15
n_classes = 2

#%%

def accuracyPerso(classifier, X_test, Y_test):
    
    Y_pred = np.argmax(classifier.predict(X_test), axis=1)
    nb_correct_pred = np.sum(Y_pred == Y_test)
    print('\nAccuracy of the model on test data: {:4.2f}%'.format(nb_correct_pred/Y_test.shape[0] * 100))
    
    return nb_correct_pred, nb_correct_pred/Y_test.shape[0]

#%%
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(n_features, activation=tf.nn.relu,input_shape = (n_features,)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
#%%

model.fit(X_train, Y_train, epochs=50,verbose=1)
#model.evaluate(X_test, Y_test)
#loss_test, accuracy_test = model.evaluate(X_test, Y_test)
nb_correct_pred,accuracy_test=accuracyPerso(model, X_test, Y_test)
#from sklearn.metrics import f1_score
#%%
classifier = KerasClassifier(model=model, clip_values=(0, 1))
#accuracyPerso(classifier, X_test, Y_test)
#%%

# =============================================================================
# #Create a ART FastGradientMethod attack.
# attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.8)
# 
# X_test_adv = attack_fgsm.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso(classifier, X_test_adv, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_fgsm * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# =============================================================================
#%% 
# =============================================================================
# #Create a ART Carlini&Wagner Infinity-norm attack.
# attack_cw = CarliniLInfMethod(classifier=classifier, eps=0.9, max_iter=100, learning_rate=0.01)
# 
# X_test_adv = attack_cw.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_cw,accuracy_test_cw=accuracyPerso(classifier, X_test_adv, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_cw * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# =============================================================================
#%%

# =============================================================================
# #Create a ART BasicIterativeMethod attack.
# attack_bim = BasicIterativeMethod(classifier, eps=0.9, eps_step=0.01, max_iter=40)
# 
# X_test_adv =attack_bim.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_bim,accuracy_test_bim=accuracyPerso(classifier, X_test_adv, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_bim * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# =============================================================================
#%%

# =============================================================================
# #Create a ART ProjectedGradientDescent attack.
# #ProjectedGradientDescent
# attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=0.8, eps_step=0.01, max_iter=100)
# 
# X_test_adv =attack_pgd.generate(X_test)
# 
# #loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
# nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso(classifier, X_test_adv, Y_test)
# perturbation = np.mean(np.abs((X_test_adv - X_test)))
# print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_pgd * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# =============================================================================
#%%

attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.8)
# attack_cw = CarliniLInfMethod(classifier=classifier, eps=0.9, max_iter=100, learning_rate=0.01)
attack_bim = BasicIterativeMethod(classifier, eps=0.9, eps_step=0.01, max_iter=40)
attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=0.8, eps_step=0.01, max_iter=100)

eps_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5,10]
nb_correct_attack_fgsm = []
# nb_correct_attack_cw = []
nb_correct_attack_bim = []
nb_correct_attack_pgd = []

average_perturbation_fgsm = []
# average_perturbation_cw = []
average_perturbation_bim = []
average_perturbation_pgd = []

for eps in eps_range:
    attack_fgsm.set_params(**{'eps': eps})
    # attack_cw.set_params(**{'eps': eps})
    attack_bim.set_params(**{'eps': eps})
    attack_pgd.set_params(**{'eps': eps})
    X_test_adv_fgsm = attack_fgsm.generate(X_test)
    # X_test_adv_cw = attack_cw.generate(X_test)
    X_test_adv_bim = attack_bim.generate(X_test)
    X_test_adv_pgd = attack_pgd.generate(X_test)
    
    nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso(classifier, X_test_adv_fgsm, Y_test)
    nb_correct_attack_fgsm += [nb_correct_pred_fgsm]
    average_perturbation_fgsm += [np.mean(np.abs((X_test_adv_fgsm - X_test)))]
    
    # nb_correct_pred_cw,accuracy_test_cw=accuracyPerso(classifier, X_test_adv_cw, Y_test)
    # nb_correct_attack_cw += [nb_correct_pred_cw]
    # average_perturbation_cw += [np.mean(np.abs((X_test_adv_cw - X_test)))]
    
    nb_correct_pred_bim,accuracy_test_bim=accuracyPerso(classifier, X_test_adv_bim, Y_test)
    nb_correct_attack_bim += [nb_correct_pred_bim]
    average_perturbation_bim += [np.mean(np.abs((X_test_adv_bim - X_test)))]
    
    nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso(classifier, X_test_adv_pgd, Y_test)
    nb_correct_attack_pgd += [nb_correct_pred_pgd]
    average_perturbation_pgd += [np.mean(np.abs((X_test_adv_pgd - X_test)))]
    
eps_range = [0] + eps_range
nb_correct_attack_fgsm = [nb_correct_pred] + nb_correct_attack_fgsm
# nb_correct_attack_cw = [nb_correct_pred] + nb_correct_attack_cw
nb_correct_attack_bim = [nb_correct_pred] + nb_correct_attack_bim 
nb_correct_attack_pgd = [nb_correct_pred] + nb_correct_attack_pgd 
average_perturbation_fgsm = [0]+ average_perturbation_fgsm
# average_perturbation_cw = [0]+ average_perturbation_cw
average_perturbation_bim = [0]+ average_perturbation_bim
average_perturbation_pgd = [0]+ average_perturbation_pgd



#%%

fig, ax = plt.subplots()
ax.plot(np.array(eps_range), np.array(nb_correct_attack_fgsm), 'b--', label='FGSM attack')
#ax.plot(np.array(eps_range), np.array(nb_correct_attack_cw), 'r--', label='C&W attack')
ax.plot(np.array(eps_range), np.array(nb_correct_attack_bim), 'g--', label='BIM attack')
ax.plot(np.array(eps_range), np.array(nb_correct_attack_pgd), 'y--', label='PGD attack')

legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#00FFCC')

plt.xlabel('Attack strength (eps)')
plt.ylabel('Correct predictions')
plt.show()

#%%
fig, ax = plt.subplots()
ax.plot(np.array(eps_range), np.array(average_perturbation_fgsm), 'b--', label='Avg Pert FGSM')
#ax.plot(np.array(eps_range), np.array(average_perturbation_cw), 'r--', label='Avg Pert C&W')
ax.plot(np.array(eps_range), np.array(average_perturbation_bim), 'g--', label='Avg Pert BIM')
ax.plot(np.array(eps_range), np.array(average_perturbation_pgd), 'y--', label='Avg Pert PGD')

legend = ax.legend(loc='lower right', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#00FFCC')

plt.xlabel('Attack strength (eps)')
plt.ylabel('Average perturbation')
plt.show()

#%%
Y_pred=np.argmax(classifier.predict(X_test_adv_pgd), axis=1)

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

precision, recall, fscore, support = score(Y_test, Y_pred)
CM=confusion_matrix(Y_test.to_numpy(), Y_pred)
print('PGD Attacks: eps= {}'.format(eps))
print('confusion_matrix:\n {}'.format(CM))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


#%%
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')
attack_names=['FGSM','BIM','PGD']
X_test_adv_attacks=[X_test_adv_fgsm,X_test_adv_bim,X_test_adv_pgd]
for attack_name,X_test_adv_attack in zip(attack_names,X_test_adv_attacks):

    Y_pred=np.argmax(classifier.predict(X_test_adv_attack), axis=1)
    fpr, tpr, threshold = roc_curve(Y_test.ravel(), Y_pred.ravel())
    
    plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(attack_name, auc(fpr, tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend();
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%