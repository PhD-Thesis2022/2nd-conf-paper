
from sklearn.ensemble import AdaBoostClassifier

import numpy as np
from matplotlib import pyplot as plt

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data = pd.read_csv('D:\Python\Jupyter\(400-400 Balanced)(Preprocessed)UNSW_2018_IoT_Botnet_Final_10_Best.csv')
data
normalized_data=(data-data.mean())/data.std()
normalized_data['attack']=data['attack']
train, test = train_test_split(normalized_data, test_size = 0.25, stratify = data['attack'], random_state = 42)
X_train = train[[ 'proto', 'saddr', 'sport','daddr','dport','seq','stddev','N_IN_Conn_P_SrcIP', 'min', 'state_number', 'mean','N_IN_Conn_P_DstIP', 'drate', 'srate', 'max']]
y_train = train.attack
X_test = test[[ 'proto', 'saddr', 'sport','daddr','dport','seq','stddev','N_IN_Conn_P_SrcIP', 'min', 'state_number', 'mean','N_IN_Conn_P_DstIP', 'drate', 'srate', 'max']]
y_test = test.attack
# %%
names = ["Nearest Neighbors",
         "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
#%%

def get_adversarial_examples(x_train, y_train):
    
    # Create and fit AdaBoostClassifier
    model = AdaBoostClassifier()
    model.fit(X=x_train, y=y_train)

    # Create ART classfier for scikit-learn AdaBoostClassifier
    art_classifier = SklearnClassifier(model=model)

    # Create ART Zeroth Order Optimization attack
    zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=20,
                    binary_search_steps=10, initial_const=1e-3, abort_early=True, use_resize=False, 
                    use_importance=False, nb_parallel=1, batch_size=1, variable_h=0.2)

    # Generate adversarial samples with ART Zeroth Order Optimization attack
    x_train_adv = zoo.generate(x_train)

    return x_train_adv, model
#%%    
def plot_results(model, x_train, y_train, x_train_adv, num_classes):
    
    fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))

    colors = ['orange', 'blue', 'green']

    for i_class in range(num_classes):

        # Plot difference vectors
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].plot([x_1_0, x_2_0], [x_1_1, x_2_1], c='black', zorder=1)

        # Plot benign samples
        for i_class_2 in range(num_classes):
            axs[i_class].scatter(x_train[y_train == i_class_2][:, 0], x_train[y_train == i_class_2][:, 1], s=20,
                                 zorder=2, c=colors[i_class_2])
        axs[i_class].set_aspect('equal', adjustable='box')

        # Show predicted probability as contour plot
        h = .01
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z_proba = Z_proba[:, i_class].reshape(xx.shape)
        im = axs[i_class].contourf(xx, yy, Z_proba, levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   vmin=0, vmax=1)
        if i_class == num_classes - 1:
            cax = fig.add_axes([0.95, 0.2, 0.025, 0.6])
            plt.colorbar(im, ax=axs[i_class], cax=cax)

        # Plot adversarial samples
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].scatter(x_2_0, x_2_1, zorder=2, c='red', marker='X')
        axs[i_class].set_xlim((x_min, x_max))
        axs[i_class].set_ylim((y_min, y_max))

        axs[i_class].set_title('class ' + str(i_class))
        axs[i_class].set_xlabel('feature 1')
        axs[i_class].set_ylabel('feature 2')
        
#%%
X_trainR=X_train.to_numpy()[:, [5, 6]]
X_testR=X_test.to_numpy()[:, [5, 6]]
y_train1=y_train.to_numpy()
y_test1=y_test.to_numpy()
y_train1
X_train_adv3, model3 = get_adversarial_examples(X_trainR, y_train1)

#%%
def test_adversarial_examples(X_train, y_train,X_test ,y_test ,clf ,name ):
    
    # Create and fit AdaBoostClassifier
    clf.fit(X=X_train, y=y_train)
    print(clf)
    # Create ART classfier for scikit-learn AdaBoostClassifier
    art_classifier = SklearnClassifier(model=clf)
    print(art_classifier)
    print('hello3\n\n\n')
    # Create ART Zeroth Order Optimization attack
    zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=20,
                    binary_search_steps=10, initial_const=1e-3, abort_early=True, use_resize=False, 
                    use_importance=False, nb_parallel=1, batch_size=1, variable_h=0.2)

    # Generate adversarial samples with ART Zeroth Order Optimization attack
    X_train_adv = zoo.generate(X_train)
    
    clf.fit(X_train_adv, y_train)
    prediction=clf.predict(X_test)
    print('The accuracy of', name ,'is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
#%%       
#for name, clf in zip(names, classifiers):
    
#    test_adversarial_examples(X_trainR, y_train1,X_testR ,y_test1 ,clf ,name )
#%%
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
#%%
from sklearn import linear_model
reg = linear_model.LinearRegression()  
reg2=linear_model.Ridge(alpha=.5) 
reg3 = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))    
reg4 = linear_model.Lasso(alpha=0.1)    
reg5 = linear_model.LassoLars(alpha=.1)         
#%%    
classifiers
test_adversarial_examples(X_trainR, y_train1,X_testR ,y_test1 ,AdaBoostClassifier(),'AdaBoost' ) 
#test_adversarial_examples(X_trainR, y_train1,X_testR ,y_test1 ,SVC(kernel="linear", C=0.025) ,'SVC' )
#test_adversarial_examples(X_trainR, y_train1,X_testR ,y_test1 ,DecisionTreeClassifier(max_depth=5) ,'DecisionTreeClassifier' )  
#%%
