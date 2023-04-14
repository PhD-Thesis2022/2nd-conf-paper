# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:45:26 2020

@author: Adam
"""
#%%

import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, CarliniL2Method,CarliniLInfMethod 
from art.attacks.evasion import BasicIterativeMethod,ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%%
filename = "D:/Python/Spyder/NSL KDD dataset/KDDTrain+.dat"
df = pd.read_csv(filename,low_memory=False,header=None)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

n_features = 41
n_classes = 2

X = df[df.columns[:-1]] 
# 0 non attack ----- 1 attack
y = df[df.columns[-1]]-1 # instead of 1 and 2 --> 0 normal , 1 attack

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

model.fit(X_train, Y_train, epochs=50,verbose=1)
classifier = KerasClassifier( model=model,)


#%%
x_test_pred = np.argmax(classifier.predict(X_test), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(Y_test, axis=1))

print("Original test data :")
print("Correctly classified: {}".format(nb_correct_pred))
print("Incorrectly classified: {}".format(Y_test.shape[0]-nb_correct_pred))
#%%

attacker = FastGradientMethod(classifier, eps=2)
x_test_adv = attacker.generate(X_test) 

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


robust_classifier = KerasClassifier(model=robust_classifier_model, use_logits=False)

#%%
#Create a ART FastGradientMethod attack.
attack_fgsm = FastGradientMethod(estimator=classifier, eps=10 )

X_test_adv = attack_fgsm.generate(X_test)

#loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso(classifier, X_test_adv, Y_test)
perturbation = np.mean(np.abs((X_test_adv - X_test)))
print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_fgsm * 100))
print('Average perturbation: {:4.2f}'.format(perturbation))

#%% 

#Create a ART Carlini&Wagner Infinity-norm attack.
attack_cw = CarliniLInfMethod(classifier=classifier)#,binary_search_steps = 100, max_iter=100)
#CarliniLInfMethod
#CarliniL2Method
X_test_adv = attack_cw.generate(X_test)

#loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
nb_correct_pred_cw,accuracy_test_cw=accuracyPerso(classifier, X_test_adv, Y_test)
perturbation = np.mean(np.abs((X_test_adv - X_test)))
print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_cw * 100))
print('Average perturbation: {:4.2f}'.format(perturbation))

#%%


#Create a ART BasicIterativeMethod attack.
attack_bim = BasicIterativeMethod(classifier, eps=5)#, eps_step=0.01, max_iter=400)

X_test_adv =attack_bim.generate(X_test)

#loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
nb_correct_pred_bim,accuracy_test_bim=accuracyPerso(classifier, X_test_adv, Y_test)
perturbation = np.mean(np.abs((X_test_adv - X_test)))
print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_bim * 100))
print('Average perturbation: {:4.2f}'.format(perturbation))
#%%

#Create a ART ProjectedGradientDescent attack.
#ProjectedGradientDescent
attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=10)#, eps_step=0.01, max_iter=250)

X_test_adv =attack_pgd.generate(X_test)

#loss_test, accuracy_test = model.evaluate(X_test_adv, Y_test)
nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso(classifier, X_test_adv, Y_test)
perturbation = np.mean(np.abs((X_test_adv - X_test)))
print('\nAccuracy on adversarial test data: {:4.2f}%'.format(accuracy_test_pgd * 100))
print('Average perturbation: {:4.2f}'.format(perturbation))
#%%
eps_range1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9, 2]

attack_fgsm = FastGradientMethod(estimator=classifier)
attack_cw = CarliniLInfMethod(classifier=classifier)
attack_bim = BasicIterativeMethod(classifier)
attack_pgd = ProjectedGradientDescent(estimator=classifier)
nb_correct_attack_fgsm = []
nb_correct_attack_cw = []
nb_correct_attack_bim = []
nb_correct_attack_pgd = []

average_perturbation_fgsm = []
average_perturbation_cw = []
average_perturbation_bim = []
average_perturbation_pgd = []

for eps in eps_range1:
    print("eps=",eps)
    attack_fgsm.set_params(**{'eps': eps})
    attack_cw.set_params(**{'eps': eps})
    attack_bim.set_params(**{'eps': eps})
    attack_pgd.set_params(**{'eps': eps})
    X_test_adv_fgsm = attack_fgsm.generate(X_test)
    X_test_adv_cw = attack_cw.generate(X_test)
    X_test_adv_bim = attack_bim.generate(X_test)
    X_test_adv_pgd = attack_pgd.generate(X_test)
    
    nb_correct_pred_fgsm,accuracy_test_fgsm=accuracyPerso(classifier, X_test_adv_fgsm, Y_test)
    nb_correct_attack_fgsm += [nb_correct_pred_fgsm]
    average_perturbation_fgsm += [np.mean(np.abs((X_test_adv_fgsm - X_test)))]
    
    nb_correct_pred_cw,accuracy_test_cw=accuracyPerso(classifier, X_test_adv_cw, Y_test)
    nb_correct_attack_cw += [nb_correct_pred_cw]
    average_perturbation_cw += [np.mean(np.abs((X_test_adv_cw - X_test)))]
    
    nb_correct_pred_bim,accuracy_test_bim=accuracyPerso(classifier, X_test_adv_bim, Y_test)
    nb_correct_attack_bim += [nb_correct_pred_bim]
    average_perturbation_bim += [np.mean(np.abs((X_test_adv_bim - X_test)))]
    
    nb_correct_pred_pgd,accuracy_test_pgd=accuracyPerso(classifier, X_test_adv_pgd, Y_test)
    nb_correct_attack_pgd += [nb_correct_pred_pgd]
    average_perturbation_pgd += [np.mean(np.abs((X_test_adv_pgd - X_test)))]
    
eps_range1 = [0] + eps_range1
nb_correct_attack_fgsm = [nb_correct_pred] + nb_correct_attack_fgsm
nb_correct_attack_cw = [nb_correct_pred] + nb_correct_attack_cw
nb_correct_attack_bim = [nb_correct_pred] + nb_correct_attack_bim 
nb_correct_attack_pgd = [nb_correct_pred] + nb_correct_attack_pgd 
average_perturbation_fgsm = [0]+ average_perturbation_fgsm
average_perturbation_cw = [0]+ average_perturbation_cw
average_perturbation_bim = [0]+ average_perturbation_bim
average_perturbation_pgd = [0]+ average_perturbation_pgd

#%%

fig, ax = plt.subplots()
ax.plot(np.array(eps_range1), np.array([x / Y_test.shape[0] for x in nb_correct_attack_fgsm]), 'y--', label='FGSM attack')
ax.plot(np.array(eps_range1), np.array([x / Y_test.shape[0] for x in nb_correct_attack_cw]), 'k--', label='C&W attack')
ax.plot(np.array(eps_range1), np.array([x / Y_test.shape[0] for x in nb_correct_attack_bim]), 'r+', label='BIM attack')
ax.plot(np.array(eps_range1), np.array([x / Y_test.shape[0] for x in nb_correct_attack_pgd]), 'b--', label='PGD attack')

legend = ax.legend(loc='best', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#00FFCC')

plt.xlabel('Attack strength (eps)')
plt.ylabel('Accuracy')
plt.show()

#%%
fig, ax = plt.subplots()
ax.plot(np.array(eps_range1), np.array(average_perturbation_fgsm), 'y*', label='Avg Pert FGSM')
ax.plot(np.array(eps_range1), np.array(average_perturbation_cw), 'k*', label='Avg Pert C&W')
ax.plot(np.array(eps_range1), np.array(average_perturbation_bim), 'r+', label='Avg Pert BIM')
ax.plot(np.array(eps_range1), np.array(average_perturbation_pgd), 'b--', label='Avg Pert PGD')

legend = ax.legend(loc='lower right', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#00FFCC')

plt.xlabel('Attack strength (eps)')
plt.ylabel('Average perturbation')
plt.show()

#%%
Training_attack = BasicIterativeMethod(robust_classifier, eps=0.9)#, eps_step=0.01, max_iter=100)

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
            #print("\n j=",j,"------------------ eps_att=",eps_att)
            attacker_robust.set_params(**{'eps': eps_att})
            x_test_adv_robust = attacker_robust.generate(X_test)
            x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
            nb_correct_robust[r,i,j] = np.sum(x_test_adv_robust_pred == np.argmax(Y_test, axis=1))/Y_test.shape[0]
            
#%%

eps_range = [0.1, 0.3, 0.5, 0.7, 0.9,1.1,1.3, 1.5,1.7, 2.0]
attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=10)
nb_correct_attack_pgd_std = []


for eps in eps_range:
    print("eps=",eps)

    attack_pgd.set_params(**{'eps': eps})

    X_test_adv_pgd = attack_pgd.generate(X_test)
     
    x_test_pred = np.argmax(classifier.predict(X_test_adv_pgd), axis=1)
    nb_correct_attack_pgd_std += [np.sum(x_test_pred == np.argmax(Y_test, axis=1))/Y_test.shape[0]]
    
#%%
#del_nb_correct_robust = nb_correct_robust[:,:-1,:-1]
import matplotlib.pyplot as plt
for r,ratio_val in enumerate(ratio_range):
    fig, ax = plt.subplots()
    
    ax.plot(np.array(eps_range), np.array(nb_correct_original[r,:]), 'g--', label='Clean test data')
    
    for i,eps_def in enumerate(eps_range):
        label='eps_def='+ str(eps_def)
        ax.plot(np.array(eps_range), np.array(nb_correct_robust[r,i,:]), '--', label=label)
        
    ax.plot(np.array(eps_range), np.array(nb_correct_attack_pgd_std), 'k--', label='PGD attack \nwithout adv training')
    legend = ax.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    ax.legend(bbox_to_anchor=(1, 1))
    title_name='Ratio of adversarial training samples ='+str(ratio_val)
    plt.title(title_name)
    plt.xlabel('Attack strength (eps)')
    plt.ylabel('Accuracy')
    plt.show()
#%%    

eps_range2=[0,	0.1,		0.3,		0.5,		0.7,		0.9,		1.1      ]
a=[0.996070649, 0.992459,	0.955586,	0.445565,	0.155785,	0.0992657,	0.0810875]
b=[0.996070649, 0.990832,	0.984124,	0.825481,	0.315658,	0.12566,	0.0876364]
c=[0.996070649, 0.981187,	0.977813,	0.973209,	0.87184,	0.369081,	0.212582]
d=[0.996070649, 0.976305,	0.975471,	0.972693,	0.967692,	0.847628,	0.495773]
e=[0.996070649,	0.702877555	,0.089343124,	0.088589006,	0.088509625,	0.088509625,	0.088509625]







fig, ax = plt.subplots()
ax.plot(np.array(eps_range2), np.array(a), 'y--', label= r'$\epsilon_{defense} = 0.1 $')
ax.plot(np.array(eps_range2), np.array(b), 'r--', label=r'$\epsilon_{defense} = 0.3 $')
ax.plot(np.array(eps_range2), np.array(c), 'b--', label=r'$\epsilon_{defense} = 0.5 $')
ax.plot(np.array(eps_range2), np.array(d), 'g--', label=r'$\epsilon_{defense} = 0.7 $')
ax.plot(np.array(eps_range2), np.array(e), 'k--', label='PGD attack \nwithout \nadv training')
legend = ax.legend(loc='best', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#00FFCC')
ax.legend(bbox_to_anchor=(1, 1))
plt.xlabel(r"Defense strength ($\epsilon_{defense}$)")   
plt.ylabel('Accuracy')
plt.show()
