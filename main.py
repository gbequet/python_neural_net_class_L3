import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles
from neural_net import NeuralNet
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json

# wsh karim j'espere t'as python3.7
# pour executer :
# python3.7 main.py
# il faut que tu entraine un reseau avant toute chose 
# ll

# fonction qui pretraitre les donnees
def pretraitement():
    root = './'
    gaussian_df = pd.read_csv(root + 'gaussian_data.csv')

    test_final_df = gaussian_df.sample(frac=0.2, random_state=42)
    gaussian_df = gaussian_df.drop(test_final_df.index)

    df_columns = gaussian_df.columns.values.tolist()

    # separation etiquettes y et attribut X 
    features = df_columns[0:2]
    label = df_columns[2:]

    X = gaussian_df[features]
    y = gaussian_df[label]

    X_test_final = test_final_df[features]
    y_test_final = test_final_df[label]

    # encodage one-hot pour les etiquettes
    y = pd.get_dummies(y)
    y_test_final = pd.get_dummies(y_test_final)

    return X,y,X_test_final,y_test_final


def analyse(model):
    y_pred = []

    for i in range (X_test_final.shape[0]):
        y_pred.append(model.predict(X_test_final.to_numpy().T[:,[i]]))
    
    return y_pred


def max_row(row):
    return np.where(row == np.amax(row))[0][0]


def ratio(y_pred, y_actual):
    cpt = 0
    for i in range(len(y_pred)):
        prediction = max_row(y_pred[i])
        attendu = max_row(y_actual[i])
        if(prediction == attendu):
            cpt = cpt + 1
    print('>> ' + str(int((100*cpt)/len(y_pred))) + '% (' + str(cpt) + '/' + str(len(y_pred)) + ')')

    return int((100*cpt)/len(y_pred))


# wsh ca enregistre le reseau de neurone 
# tu a du en entrainer un avant !
def save_best(nn, taux):
    w = b = []
    for i in range(len(nn.weights)-1):
        w.append(nn.weights[i].tolist())
        b.append(nn.biases[i].tolist())

    data = {}
    data['ratio'] = taux
    data['poids'] = w
    data['biais'] = b
    data['e_train'] = nn.e_train
    data['e_test'] = nn.e_test

    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)


# wsh karim c'est la fonction qui lit un reseau de neuronnes
# apres les tableaux des poids sont dans le fichier data.txt en format json
def read_best(nn):
    with open('data.txt') as json_file:
        data = json.load(json_file)
        w = b = []
        for i in range(len(nn.weights)-1):
            w.append(data['poids'][i])
            b.append(data['biais'][i])

        taux = data['ratio']

        # wsh ca recupere les matrices de 
        # poids, biais, e_train, e_test
        nn.weights = w
        nn.biases = b
        nn.e_train = data['e_train']
        nn.e_test = data['e_test']
        return taux


def make_error_graph(nn, i, w):
    plt.subplot(w)
    plt.plot(list(range(i)), nn.e_train, label='Train')
    plt.plot(list(range(i)), nn.e_test, label='Test')
    plt.xlabel('Epoque')
    plt.ylabel('Erreur moyenne')
    plt.legend()


def make_confusion(y_pred, y_actual, w):
    plt.subplot(w)
    confusion_mtx = np.zeros((3,3), dtype=int)
    for i in range(X_test_final.shape[0]):
        pred = max_row(y_pred[i])
        att = max_row(y_actual[i])
        confusion_mtx[pred][att] = confusion_mtx[pred][att] + 1

    class_names = ['class-0', 'class-1', 'class-2']
    plt.figure(figsize = (8,8))
    sns.set(font_scale=2) # label size
    ax = sns.heatmap(confusion_mtx, annot=True, annot_kws={"size": 30}, # font size
    cbar=False, cmap='Blues', fmt='d', # format (int)
    xticklabels=class_names, yticklabels=class_names)
    ax.set(title='', xlabel='Actual', ylabel='Predicted')


# MAIN
if __name__ == '__main__':
    X,y,X_test_final,y_test_final = pretraitement()
    y_actual = y_test_final.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    plt.subplot(321)

    X1, Y1 = make_gaussian_quantiles(n_samples=1500, n_features=2, n_classes=3)

    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')

    cond = True
    t = False
    while(cond):
        cmd = input('> ')


        if(cmd == 'q'):
            print('>> 👋')
            cond = False


        # wsh karim la c'est la commande pour lire dans un fichier
        # affiche les resultats du modele de "data.txt"
        elif(cmd == 'use'):
            # WSHHH ca c'est ce qui manque, recuperer 
            taux = read_best(nn)
            make_error_graph(nn, nn.besti) # ca dessine juste le graphe d'erreur
            make_confusion(y_pred, y_actual) # ca affiche la matrice de confusion


        # wsh, 
        # enregistre le modele dans un fichier
        elif(cmd == 'save'):
            y_pred = analyse(nn) # ca analyse le reseau de neurones nn
            taux = ratio(y_pred, y_actual) # on recup le taux en %

            save_best(nn, taux)


        # cherche un model a plus de n%
        elif ('sup' in cmd):
            minimum = int(cmd[4:])
            taux = 0
            t2_start = time.perf_counter()
            tmp_start = time.perf_counter()

            while (taux < minimum):
                nn = NeuralNet(X_train, y_train, X_test, y_test, (4,3,2), 'tanh')
                i = nn.early_stopping(3,3)
                y_pred = analyse(nn)
                taux = ratio(y_pred, y_actual)
                tmp_stop = time.perf_counter()
                print ('---> ' + str(int(tmp_stop-tmp_start)) + 's')
                tmp_start = time.perf_counter()

            t2_stop = time.perf_counter()
            print ('--> ' + str(int(t2_stop-t2_start)) + 's')
            make_error_graph(nn, i, 322)
            make_confusion(y_pred, y_actual, 323)

            save_best(nn, taux)
            plt.show()
            plt.clf()


        # entraine un reseau de neuronnes avec early stopping
        elif(cmd == 'll'):
            nn = NeuralNet(X_train, y_train, X_test, y_test, (4,3,2), 'tanh')
            t2_start = time.perf_counter()
            i = nn.early_stopping(5,5)
            t2_stop = time.perf_counter()

            print ('>> Temps d\'execution fit early stopping :', (t2_stop-t2_start))
            y_pred = analyse(nn)
            ratio(y_pred, y_actual)

            make_error_graph(nn, i)
            make_confusion(y_pred, y_actual)


        # entraine un reseau de neuronnes sur les donnees (200 epoques)
        elif(cmd == 'tt'):
            nn = NeuralNet(X_train, y_train, X_test, y_test, (4,3,2), 'tanh')
            t1_start = time.perf_counter()
            nn.fit()
            t1_stop = time.perf_counter()

            print ('>> Temps d\'execution fit :', (t1_stop-t1_start))
            ratio(y_pred, y_actual)
            y_pred = analyse(nn)

            make_error_graph(nn, nn.epoch)
            make_confusion(y_pred, y_actual)


        # analyse les données mise a disposition sur moodle
        elif(cmd == 'mm'):
            y_actual = np.genfromtxt('tanh_y_actual.csv', delimiter=',')
            y_pred = np.genfromtxt('tanh_y_pred.csv', delimiter=',')
            ratio(y_pred, y_actual)
            make_confusion(y_pred, y_actual)


        # ratio
        elif('r' in cmd):
            y_pred = analyse(nn)
            ratio(y_pred, y_actual)


        # predit la ieme instance de X_test_final
        elif('p' in cmd):
            ind = int(cmd[8:])
            prediction = nn.predict(X_test_final.to_numpy().T[:,[ind]])
            p = np.argmax(prediction)
            att = np.argmax(y_test_final.to_numpy().T[:,[ind]])
            print('classe attendu : ', att)
            print('classe predite : ', p)


        # aide
        else:
            print('>> commandes disponible :\n -train : entraine le ptn de reseau\n -predict i : predit la classe de la ieme instance de X_test_final\n -quit')
