import pandas as pd
from sklearn.linear_model import Perceptron

df = pd.read_csv("Social_Network_Ads.csv", sep=",")

df = df.drop(columns="User ID")

df['Male'] = (df["Gender"] == "Male").astype(int)  #agregando una nueva columna al dataset
df['Female'] = (df["Gender"] == "Female").astype(int)  #agregando una nueva columna al dataset

df = df.drop(columns='Gender')

columns_order = list(df.columns.difference(["Male", "Female", "Purchased"])) + ["Male", "Female", "Purchased"]  # Corregido aquí

df = df[columns_order]  # Aplicando el orden correcto de columnas

#Perceptron idea 
xtrain = df.iloc [:319, 0:3 ]
ytrain = df.iloc [ :319, 4: ] 

xtest = df.iloc [ 320:, 0:3]
ytest = df.iloc [ 320:, 4: ]
# print (xtrain.head(1))
# print (ytrain.head(1))
# print (xtest.head(1))
# print (ytest.head(1))
#X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(xtrain, ytrain)
Perceptron()
#clf.score(xtrain, ytrain)
#clf.score(xtest, ytest)
print (clf.score(xtrain, ytrain))