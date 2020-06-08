#!/usr/bin/env python
# coding: utf-8

# # Particle Swarm Optimzation for tuning Machine Learning model to global optima

# define your objective function aka Machine Learning model here and return 1-(value you want to maximize)

# In[1]:


def func1(x0):
    import pandas as pd
    import numpy as np
    import plotly.offline as plt
    import plotly.graph_objs as go
    import sklearn.datasets
    import sklearn.metrics
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
#<------------------------Reading the training File------------------------------------------->
    train = pd.read_csv('train.csv')
    train = train.drop(['Unnamed: 0'],axis=1)
#<------------------------OutLier Removal from the data-set-------------------------------->
    from scipy import stats
    z = np.abs(stats.zscore(train))
    threshold = 3
    original_train = train
    train = train[(z < 3).all(axis=1)]
    train.columns
    y = train['target variable']
    x = train[train.columns[:-1]]
    x = np.asarray(x)
    y = np.asarray(y)
#<--------------------------OverSampling the data-set------------------------------------->
    from imblearn.over_sampling import ADASYN 
    sm = ADASYN(random_state = 2) 
    x, y = sm.fit_sample(x, y.ravel())
    
#<----------------------------Define your Model here --------------------------------------->
    clf = RandomForestClassifier(n_estimators=int(x0[0]),bootstrap = True, max_depth = int(x0[1]), max_features = int(x0[2]), max_leaf_nodes = 17, n_jobs = 9, random_state = 42)

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, random_state=42)
    # X is the feature set and y is the target
    fold = 1
    for train_index, val_index in skf.split(x,y): 
        #print("Train:", train_index, "Validation:", val_index) 
        X_train, X_test = x[train_index], x[val_index] 
        Y_train, Y_test = y[train_index], y[val_index]
        model = clf.fit(X_train,Y_train)
        a_train = model.predict(X_train)
        a_test = model.predict(X_test)
        print('K-FOLD #',fold)
        fold+=1
        print('Train AUC-ROC Score: ',roc_auc_score(a_train,Y_train),' Val AUC-ROC score: ',roc_auc_score(a_test,Y_test),'\n')
        from sklearn.metrics import cohen_kappa_score

        print('validation kappa:-',cohen_kappa_score(a_test, Y_test))
    test = pd.read_csv('test.csv')
    test = test.drop(['Unnamed: 0'],axis=1)

    y_testt = test['Disease Status (NSCLC: primary tumors; Normal: non-tumor lung tissues)']

    x_testt = test[test.columns[:-1]]

    y_testt_pred = model.predict(x_testt)
    a_test = []
    for i in y_testt_pred:
        if i>0.5:
            a_test.append(1)
        else:
            a_test.append(0)
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import cohen_kappa_score
#<-----------------Printing the Parameters and the maximized objective value-------------------------->
    print('The Parameters are respectively:',x0)
    print('test kappa:-',cohen_kappa_score(y_testt, a_test))
#<-----------------Return 1-(value to be maximized)--------------------------------------------------->
    return 1-cohen_kappa_score(y_testt, a_test)


# Particle Swarm Optimization (PSO) starts here

# In[2]:


from __future__ import division
import random
import math


#--- MAIN 
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.1       # constant inertia weight (how much to weigh the previous velocity)
        c1=7      # cognative constant
        c2=10        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions
        
        num_dimensions=len(x0)
        
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))
            

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
#                 print('j')
                swarm[j].evaluate(costFunc)
                

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        print ('FINAL:')
        print (pos_best_g)
        print (err_best_g)

if __name__ == "__PSO__":
    main()
    

initial=[10,4,8]               # initial starting location [x1,x2...]
bounds=[(2,20),(2,20),(2,20)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func1,initial,bounds,num_particles=3,maxiter=10)

