import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
import re
from utility import Utility
from sklearn.utils import shuffle
# from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
import statistics


class NeuralNet:

    def __init__(self, X_train = None, y_train = None, X_test = None, y_test = None, 
    hidden_layer_sizes=(4,), activation='identity', learning_rate=0.01, epoch=200):
        self.nb_layers = len(hidden_layer_sizes)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.weights = []
        self.biases = []
        self.Z = []
        self.A = []
        self.df = []
        self.e_train = []
        self.e_test = []

        if(activation == 'identity'):
            self.activation = Utility.identity
        if(activation == 'tanh'):
            self.activation = Utility.tanh
        if(activation == 'relu'):
            self.activation = Utility.relu
        if(activation == 'sigmoid'):
            self.activation = Utility.sigmoid
        if(activation == 'softmax'):
            self.activation = Utility.softmax

        self.__weights_initialization(X_train, y_train, hidden_layer_sizes)


    # initialise les matrices de poids et de biais pour chaque couche du reseau
    def __weights_initialization(self, X, y, hidden_layer_sizes):
        self.weights.append(np.random.uniform(-1,1, (hidden_layer_sizes[0],X.shape[1])))
        self.biases.append(np.random.uniform(-1,1, (hidden_layer_sizes[0],1)))
    
        for i in range(1, self.nb_layers):
            self.weights.append(np.random.uniform(-1,1, (hidden_layer_sizes[i],hidden_layer_sizes[i-1])))
            self.biases.append(np.random.uniform(-1,1, (hidden_layer_sizes[i],1)))

        self.weights.append(np.random.uniform(-1,1, (y.shape[1],hidden_layer_sizes[-1])))
        self.biases.append(np.random.uniform(-1,1, (y.shape[1],1)))



    # passe avant
    def __forward_pass(self, X, y):
        # premiere couche :
        self.Z.append(np.dot(self.weights[0],X) + self.biases[0])
        a,df = self.activation(self.Z[0])
        self.A.append(a)
        self.df.append(df)

        for i in range(1,self.nb_layers):
            self.Z.append(np.dot(self.weights[i],self.A[i-1]) + self.biases[i])
            a,df = self.activation(self.Z[i])
            self.A.append(a)
            self.df.append(df)
        
        # derniere couche :
        self.Z.append(np.dot(self.weights[self.nb_layers],self.A[self.nb_layers-1]) + self.biases[self.nb_layers])
        self.A.append(Utility.softmax(self.Z[self.nb_layers])) # activée par softmax

        # calcul de l'erreur entre y_hat et y
        ce = Utility.cross_entropy_cost(self.A[-1],y)
        # ce = Utility.MSE_cost(self.A[-1],y)

        # prediction = np.argmax(self.A[-1])

        return ce, self.A[-1]


    # passe arriere
    def __backward_pass(self, X, y):
        delta = [None] * (self.nb_layers + 1)
        dW = [None] * (self.nb_layers + 1)
        db = [None] * (self.nb_layers + 1)

        # derniere couche :
        delta[-1] = self.A[-1] - y
        dW[-1] = np.dot(delta[-1],self.A[-1].T)
        db[-1] = delta[-1]

        for i in range(self.nb_layers-1,-1,-1):
            delta[i] = np.multiply(np.dot(self.weights[i+1].T, delta[i+1]), self.df[i])

            if i==0:
                dW[i] = np.dot(delta[i], X.T)
            else:
                dW[i] = np.dot(delta[i], self.A[i-1].T)

            db[i] = delta[i]

        # mise a jour des poids et des biais en fonction des matrices de derivees dW et db
        for i in range(len(self.weights)-1):
            self.biases[i] = self.biases[i] - self.learning_rate*db[i]
            self.weights[i] = self.weights[i] - self.learning_rate*dW[i]

        # remise a zero des activations, entrees, derivées de chaque couche
        # pour la prochaine passe avant
        self.A = []
        self.Z = []
        self.df = []

    
    def __epoch(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        erreurs_train = []
        erreurs_test = []
        size_train = self.X_train.shape[0]
        size_test = self.X_test.shape[0]

        for i in range(size_train):
            X = self.X_train.to_numpy().T[:,[i]]
            y = self.y_train.to_numpy().T[:,[i]]
            err, a = self.__forward_pass(X,y)
            erreurs_train.append(err)
            self.__backward_pass(X,y)
        erreur_train = sum(erreurs_train)/len(erreurs_train)

        for i in range(size_test):
            X = self.X_test.to_numpy().T[:,[i]]
            y = self.y_test.to_numpy().T[:,[i]]
            err, a = self.__forward_pass(X,y)
            erreurs_test.append(err)
            self.A = []
            self.Z = []
            self.df = []
        erreur_test = sum(erreurs_test)/len(erreurs_test)

        self.e_train.append(erreur_train)
        self.e_test.append(erreur_test)


    def early_stopping(self, n, p):
        v = 1000
        i = j = besti = tmp = 0
        best_weights = self.weights
        best_biases = self.biases

        while (j < p):
            for k in range(n):
                self.__epoch()
            
            i = i+n
            vbis = self.e_test[-1]
            if (vbis < v):
                j = 0
                best_weights = self.weights
                best_biases = self.biases
                v = vbis
            else:
                j = j+1
            besti = i

        return besti


    def fit(self):
        for i in range(self.epoch):
            self.__epoch()


    def predict(self, x):
        err, a = self.__forward_pass(x,self.y_train.to_numpy().T[:,[0]])
        self.A = []
        self.Z = []
        self.df = []
        return a

    
    def test_donnees(self, X, y):
        cpt = 0
        size = X.shape[0]

        for i in range(size):
            tmpX = X.to_numpy().T[:,[i]]
            tmpy = y.to_numpy().T[:,[i]]
            lol, a = self.__forward_pass(tmpX,tmpy)
            prediction = np.argmax(self.A[-1])
            ty = np.argmax(tmpy)
            if prediction == ty:
                cpt += 1
            self.A = []
            self.Z = []
            self.df = []
        