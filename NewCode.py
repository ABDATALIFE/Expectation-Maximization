#%% Importing Libraries
import mpmath as mp
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import pickle
import decimal
import math as mm


#%% Importing Data
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

df_newest = df.sample(frac = 0.0001)
X = df_newest.values


#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_1 = X[:,0].reshape(-1,1)
X_2 = X[:,1].reshape(-1,1)
X_1 = sc_X.fit_transform(X_1)
X_2 = sc_Y.fit_transform(X_2)

X_new = np.c_[X_1,X_2]




#%%
class GMM:

    def __init__(self,X,number_of_sources,iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None
        
    
    
    
    
    

    """Define a function which runs for iterations, iterations"""
    def run(self):
        
        def bigexp(xyz):
            abc = decimal.Decimal(mm.exp(1))**xyz
            return float(abc)
        
        def multivariate_gaussian(data,miu,sigma):
            np.seterr(invalid='ignore')
            numerator= np.expm1(-0.5*(data-miu)@np.linalg.inv(sigma)@(data-miu).T)
            denominator=(2*np.pi)**(data.shape[1]/2)*np.linalg.det(sigma)**0.5

            return np.diagonal(numerator/denominator)
        
        np.seterr(invalid='ignore')
        
        self.reg_cov = np.identity(len(self.X[0]))
        x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        self.XY = np.array([x.flatten(),y.flatten()]).T
           
                    
        """ 1. Set the initial mu, covariance and pi values"""
        self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_of_sources,len(self.X[0]))) # This is a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
        self.cov = np.zeros((self.number_of_sources,len(X[0]),len(X[0]))) # We need a nxmxm covariance matrix for each source since we have m features --> We create symmetric covariance matrices with ones on the digonal
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],5)


        self.pi = np.ones(self.number_of_sources)/self.number_of_sources # Are "Fractions"
        log_likelihoods = [] # In this list we store the log likehoods per iteration and plot them in the end to check if
                             # if we have converged
        colours = ['black','red','green','blue']    
        """Plot the initial state"""    
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        ax0.scatter(sc_X.inverse_transform(self.X[:,0]),sc_Y.inverse_transform(self.X[:,1]),alpha = 0.4,color='r')
        #ax0.scatter(sc_X.inverse_transform(self.X[:,0]),sc_Y.inverse_transform(self.X[:,1]),alpha = 0.7,color='g')
        ax0.set_title('Initial state')
        for m,c,cl in zip(self.mu,self.cov,colours):
            c += self.reg_cov
            #multi_normal = multivariate_normal(mean=m,cov=c)
            ax0.contour(sc_X.inverse_transform(np.sort(self.X[:,0])),sc_Y.inverse_transform(np.sort(self.X[:,1])),sc_X.inverse_transform(multivariate_gaussian(self.XY,m,c)).reshape(len(self.X),len(self.X)),colors=cl,alpha=0.3)
            ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
        
        for i in range(self.iterations):               

            """E Step"""
            r_ic = np.zeros((len(self.X),len(self.cov)))

            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                co+=self.reg_cov
                #mn = multivariate_normal(mean=m,cov=co)
                r_ic[:,r] = p*multivariate_gaussian(self.X,m,co)/np.sum([pi_c*multivariate_gaussian(self.X,mu_c,cov_c) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)

            """
            The above calculation of r_ic is not that obvious why I want to quickly derive what we have done above.
            First of all the nominator:
            We calculate for each source c which is defined by m,co and p for every instance x_i, the multivariate_normal.pdf() value.
            For each loop this gives us a 100x1 matrix (This value divided by the denominator is then assigned to r_ic[:,r] which is in 
            the end a 100x3 matrix).
            Second the denominator:
            What we do here is, we calculate the multivariate_normal.pdf() for every instance x_i for every source c which is defined by
            pi_c, mu_c, and cov_c and write this into a list. This gives us a 3x100 matrix where we have 100 entrances per source c.
            Now the formula wants us to add up the pdf() values given by the 3 sources for each x_i. Hence we sum up this list over axis=0.
            This gives us then a list with 100 entries.
            What we have now is FOR EACH LOOP a list with 100 entries in the nominator and a list with 100 entries in the denominator
            where each element is the pdf per class c for each instance x_i (nominator) respectively the summed pdf's of classes c for each 
            instance x_i. Consequently we can now divide the nominator by the denominator and have as result a list with 100 elements which we
            can then assign to r_ic[:,r] --> One row r per source c. In the end after we have done this for all three sources (three loops)
            and run from r==0 to r==2 we get a matrix with dimensionallity 100x3 which is exactly what we want.
            If we check the entries of r_ic we see that there mostly one element which is much larger than the other two. This is because
            every instance x_i is much closer to one of the three gaussians (that is, much more likely to come from this gaussian) than
            it is to the other two. That is practically speaing, r_ic gives us the fraction of the probability that x_i belongs to class
            c over the probability that x_i belonges to any of the classes c (Probability that x_i occurs given the 3 Gaussians).
            """

            """M Step"""

            # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
            self.mu = []
            self.cov = []
            self.pi = []
            log_likelihood = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c],axis=0)
                mu_c = (1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
                self.mu.append(mu_c)

                # Calculate the covariance matrix per source based on the new mean
                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)
                # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
                self.pi.append(m_c/np.sum(r_ic)) # Here np.sum(r_ic) gives as result the number of instances. This is logical since we know 
                                                # that the columns of each row of r_ic adds up to 1. Since we add up all elements, we sum up all
                                                # columns per row which gives 1 and then all rows which gives then the number of instances (rows) 
                                                # in X --> Since pi_new contains the fractions of datapoints, assigned to the sources c,
                                                # The elements in pi_new must add up to 1

            
            
            """Log likelihood"""
            log_likelihoods.append(np.log1p(np.sum([k*multivariate_gaussian(self.X,self.mu[i],self.cov[j]) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))
            
            

            """
            This process of E step followed by a M step is now iterated a number of n times. In the second step for instance,
            we use the calculated pi_new, mu_new and cov_new to calculate the new r_ic which are then used in the second M step
            to calculat the mu_new2 and cov_new2 and so on....
            """

        fig2 = plt.figure(figsize=(10,10))
        ax1 = fig2.add_subplot(111) 
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0,self.iterations,1),log_likelihoods,linewidth=5,alpha = 1)
        ax1.set_xlim([-100,100])
        ax1.set_ylim([-50,50])
        plt.show()
    
    """Predict the membership of an unseen, new datapoint"""
    def predict(self,Y):
        
        def multivariate_gaussian(data,miu,sigma):
            np.seterr(invalid='ignore')
            numerator= np.expm1(-0.5*(data-miu)@np.linalg.inv(sigma)@(data-miu).T)
            denominator=(2*np.pi)**(data.shape[1]/2)*np.linalg.det(sigma)**0.5

            return np.diagonal(numerator/denominator)
        
        colours = ['black','red','green','blue']
        # PLot the point onto the fittet gaussians
        fig3 = plt.figure(figsize=(10,10))
        ax2 = fig3.add_subplot(111)
        ax2.scatter(sc_X.inverse_transform(self.X[:,0]),sc_Y.inverse_transform(self.X[:,1]))
        for m,c,cl in zip(self.mu,self.cov,colours):
            #multi_normal = multivariate_normal(mean=m,cov=c)
            ax2.contour(sc_X.inverse_transform(np.sort(self.X[:,0])),sc_Y.inverse_transform(np.sort(self.X[:,1])),multivariate_gaussian(self.XY,m,c).reshape(len(self.X),len(self.X)),colors=cl,alpha=0.6)
            ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
            ax2.set_title('Final state')
            for y in Y:
                ax2.scatter(y[0],y[1],c='orange',zorder=10,s=100)
        prediction = []        
        for m,c in zip(self.mu,self.cov):  
            #print(c)
            prediction.append(multivariate_gaussian(Y,m,c)/np.sum([multivariate_gaussian(Y,mean,cov) for mean,cov in zip(self.mu,self.cov)]))
        #plt.show()
        return prediction
        
    

#%%
GMM = GMM(X_new,3,50)

#%%
GMM.run()
#%%
qwe = sc_X.fit_transform([[0]])
rty = sc_X.fit_transform([[0]])


outputpredictors =[[qwe,rty]]
outputpredictors = np.array(outputpredictors)

values = GMM.predict(outputpredictors)