from sklearn import linear_model
from sklearn.metrics import mean_squared_error





class BaseRegressor():
    def __init__(self):
        # http://scikit-learn.org/stable/modules/linear_model.html
        self.dct = {
                'linear':linear_model.LinearRegression(),
                'Ridge':linear_model.Ridge(alpha = .5),  # linéaire avec pénalisation de complexité
                'lasso':linear_model.Lasso(alpha = 0.1)
                
                }
        self.successes = {}

    def train(self,X,Y):
        # Entrées :
            # X = numpy array (N_instances,N_features)
            # Y = numpy array (N_instances)
        # Sorties :
            # None
        
        limit = (9*X.shape[0])//10
        X_train,Y_train = X[:limit,:],Y[:limit]
        X_val,Y_val= X[limit:,:],Y[limit:]
        
        for reg_name in self.dct:
            reg = self.dct[reg_name]
            reg.fit(X_train,Y_train)
            preds = reg.predict(X_val)
            self.successes[reg_name] = 1./mean_squared_error(preds,Y_val)
            
            
    def predict(self,X):
        # Entrées :
            # X = numpy array (N_instances,N_features)
        # Sorties :
            # Y = numpy array (N_instances)
            
        predictions = np.zeros((X.shape[0],))
        
        for name in self.dct:
            reg = self.dct[name]
            predictions += reg.predict(X)*self.successes[name]
        
        predictions /= sum([self.successes[name]  for name in self.successes ])
        return predictions
    
    
    


"""
import numpy as np    
A = np.random.rand(100,20)    


def f(A):
    Y = []
    for i in range(A.shape[0]) :
        y = sum([j*A[i,j] for j in range(A.shape[1])])
        Y.append(y)
    Y = np.array(Y)
    return Y

Y_train = f(A)


epsilon = 0.005*(np.random.rand(100,20) -0.5)
X_train = A + epsilon    
    

X_test = np.random.rand(100,20)   
Y_test = f(X_test)
    
R = BaseRegressor()
R.train(A,Y_train)
Y_pred = R.predict(X_test)
print(mean_squared_error(Y_pred,Y_test))


R = BaseRegressor()
R.train(X_train,Y_train)
Y_pred = R.predict(X_test)
print(mean_squared_error(Y_pred,Y_test))



"""





