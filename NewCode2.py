#%%
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import pandas as pd
import pickle




#%%


df = pickle.load(open('C:/Users/Abdul Basit Aftab/Desktop/EM_Project_2/EM Project/Project code/EM with real data/trajectories.pickle','rb'))
xList = []
yList = []
for i in range(len(df)):
    for j in range(len(df[i])):
        xList.append(int(df[i][j][0]))
    
for i in range(len(df)):
    for j in range(len(df[i])):
        yList.append(int(df[i][j][1]))

coord = np.c_[xList,yList]
df = pd.DataFrame(coord,columns=["x1","x2"])

df_newest = df.sample(frac = 0.001)
X = df_newest.values



#%%
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T


#%%
GMM = GaussianMixture(n_components=5).fit(X) # Instantiate and fit the model
print('Converged:',GMM.converged_) # Check if the model has converged
means = GMM.means_ 
covariances = GMM.covariances_



#%%
# Predict
Y = np.array([[0.5],[0.5]])
prediction = GMM.predict_proba(Y.T)
print(prediction)


#%%
# Plot   
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(X[:,0],X[:,1])
ax0.scatter(Y[0,:],Y[1,:],c='orange',zorder=10,s=100)
for m,c in zip(means,covariances):
    multi_normal = multivariate_normal(mean=m,cov=c)
    ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
    ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    
plt.show()
