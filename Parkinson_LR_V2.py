#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
import warnings
from pylab import savefig
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
warnings.filterwarnings( "ignore" )
  
# Logistic Regression
class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
    #Training function
    def fit( self, X, Y ) :              
        self.m, self.n = X.shape        
        # Weight initialization        
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        #Gradient descent function          
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    #Gradient descent update values functiom
    def update_weights( self ) :           
        A = 1/(1+np.exp(-(self.X.dot(self.W)+self.b)))
          
        # calculate gradients        
        tmp = (A-self.Y.T)        
        tmp = np.reshape(tmp,self.m)        
        dW = np.dot(self.X.T,tmp)/self.m         
        db = np.sum(tmp)/self.m 
          
        #Update weight values  
        self.W = self.W-self.learning_rate*dW    
        self.b = self.b-self.learning_rate*db
          
        return self

    #Predict function  
    def predict( self, X ) :    
        Z = 1/(1+np.exp(-(X.dot(self.W)+self.b)))        
        Y = np.where(Z>0.5,1,0)        
        return Y
  
def main():
      
    # Import data base
    path = (r"C:\Virtual Environment\Scripts\Data - Parkinsons.csv")
    df = pd.read_csv(path)

    #Check correllations
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(df.corr(),annot=True,linewidth=0.05,ax=ax, fmt= '.2f', cmap=plt.cm.cool)   
    plt.savefig('BC_HM.png', dpi=400)
    plt.show()
    print("I will delete variables with more than one value of 1.0 or close (light pink) and close to 0 (light blue), this means that it may be a problem for my result. This also means \nmy model would not be able to distinguish between independent and dependent variables")
    
    #Count values
    print("\nCounting positive and negative values:")
    print(df['status'].value_counts())

    #Set x and y
    X = df.drop(['name','status'], axis = 1)
    Y = df.status.values

    #Split into test (32%) and train (68%) for hand LR
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .32, random_state = 30 )
    #Split into test (20%) and train (80%) for framework LR
    x_train, x_test, y_train, y_test = train_test_split(X , Y, test_size = .20, random_state = 42)

    #Perform logistic regression with framework
    log_reg = LogisticRegression(solver='liblinear', max_iter=5000)
    log_reg.fit(x_train,y_train)

    print('\nCalculated framework coeficients are:') #m's
    print(log_reg.coef_)
    print('\nCalculated framework intercept:') #b
    print(log_reg.intercept_)

   # define oversampling strategy
    SMOTE1 = SMOTE()
    # fit and apply the transform
    X_train_SMOTE, y_train_SMOTE = SMOTE1.fit_resample(X_train, Y_train)

    # Model training by hand    
    model = LogitRegression(learning_rate = 0.0001, iterations = 5000)

    model.fit( X_train_SMOTE,y_train_SMOTE )   
      
    # Prediction on test set for hand prediction
    Y_pred = model.predict( X_test ) 

    #Make predictions on test set for framework
    y_pred = log_reg.predict(x_test)
      
    # measure hand performance    
    correctly_classified = 0    
      
    # counter    
    count = 0    
    for count in range( np.size( Y_pred ) ) :  
        
        if Y_test[count] == Y_pred[count] :            
            correctly_classified = correctly_classified + 1
            count = count + 1

    #True possitive(Upper-left)
    #True negative(lower right)
    #False positive(top-right)
    #False negative(lower-left)
    print("\nConfusion matrix results for hand test model 32%")
    print(confusion_matrix(Y_test,Y_pred))             
    print("\nAccuracy on test set by hand model:  ", (correctly_classified / count ) * 100 )

    print("\nConfusion matrix results for framework test model 20%")
    print(confusion_matrix(y_test,y_pred))
    print("Accuracy score for Logistic Regresion with framework is %f" %(accuracy_score(y_test, y_pred)))   
    
    #UserTest
    print("\nAnswer the next questions: \n")
    UserInfo = []
    UserInfo.append(float(input("Average vocal fundamental frequency (Healthy Average 181.9377708): ")))
    UserInfo.append(float(input("Maximum vocal fundamental frequency (Healthy Average 223.63675): ")))
    UserInfo.append(float(input("Minimum vocal fundamental frequency (Healthy Average 145.207292): ")))
    UserInfo.append(float(input("Dysphonic Voice Pattern percetage (Healthy Average 0.00386604): ")))
    UserInfo.append(float(input("Relative Amplitude Perturbation (Healthy Average 0.000023375): ")))
    UserInfo.append(float(input("Absolute Dysphonic Voice Pattern (Healthy Average 0.001925: ")))
    UserInfo.append(float(input("Point Period Perturbation (Healthy Average 0.002056042): ")))
    UserInfo.append(float(input("DDP (Healthy Average 0.005776042): ")))
    UserInfo.append(float(input("MDVP:Shimmer (Healthy Average 0.017615208): ")))
    UserInfo.append(float(input("MDVP:Shimmer(dB) (Healthy Average 0.162958333): ")))
    UserInfo.append(float(input("Shimmer:APQ3 (Healthy Average 0.009503542): ")))
    UserInfo.append(float(input("Shimmer:APQ5 (Healthy Average 0.010508542): ")))
    UserInfo.append(float(input("Shimmer Perturbation Quotient (Healthy Average 0.013304792): ")))
    UserInfo.append(float(input("Shimmer:DDA (Healthy Average 0.028511458): ")))
    UserInfo.append(float(input("Noice to Armonics Ratio (Healthy Average 0.01148271): ")))
    UserInfo.append(float(input("HNR (Healthy Average 24.67875): ")))
    UserInfo.append(float(input("RPDE (Healthy Average 0.442551875): ")))
    UserInfo.append(float(input("Detrended Fluctuation Analysis (Healthy Average 0.695715563): ")))
    UserInfo.append(float(input("Fundamental Frequency Variation 1 (Healthy Average -6.759263875): ")))
    UserInfo.append(float(input("Fundamental Frequency Variation 2 (Healthy Average 0.160292): ")))
    UserInfo.append(float(input("D2 (Healthy Average 2.154490729): ")))
    UserInfo.append(float(input("PPE (Healthy Average 0.123017104): ")))

    UserInfo_df = pd.DataFrame([UserInfo])
    predictionh = model.predict(UserInfo_df)
    predictionf = log_reg.predict(UserInfo_df)

    print("\nResults will be printed with 1 as Parkinson case and 0 as healthy")
    print("\nHand Prediction: ")
    print(predictionh)
    
    print("Results will be printed with 1 as Parkinson case and 0 as healthy")
    print("Framework prediction: ")
    print(predictionf)
 
if __name__ == "__main__" :     
    main()
# %%