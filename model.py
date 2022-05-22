import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import  pickle

df = pd.read_csv('student-mat.csv', sep=';')

plt.figure(figsize=(11, 11))
sns.heatmap(df.corr().round(1), annot=True)

df = df[['absences', 'G1', 'G2', 'G3']]


X = df.iloc[:, 0:3].values

y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


reg = LinearRegression()
reg.fit(X_train, y_train)

pickle.dump(reg, open("model.pkl", "wb"))
#y_pred = reg.predict(X_test)


#pred = np.array([0, 90, 90]).reshape(1, -1)

#print(reg.score(X, y))
#print(reg.predict(pred))


