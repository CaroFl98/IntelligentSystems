#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np

#Load dataframe
path = (r"C:\Virtual Environment\Scripts\Data - Parkinsons.csv")
df = pd.read_csv(path)

#Check correllations
fig, ax = plt.subplots(figsize=(20,20))
print(sns.heatmap(df.corr(),annot=True,linewidth=0.05,ax=ax, fmt= '.2f', cmap=plt.cm.cool))
print("I will delete variables with value of 1.0, this means that it may be a \n problem for my result. This also means my model would not be able to distinguish between independent and dependent variables")

#%%
#Collinearity with multiple variables
df_x = df.drop(['MDVP:Shimmer(dB)','spread1','MDVP:Shimmer','D2','Shimmer:DDA','RPDE','Shimmer:APQ5','MDVP:Fo(Hz)',
'PPE','HNR','Shimmer:APQ3','Jitter:DDP','name','status'],axis=1)
df_y = df['status']

#%%
#Count values
print("Counting positive and negative values:")
print(df['status'].value_counts())

#%%
#Split into test (20%) and train (80%)
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.20,random_state=42)

#Perform logistic regression
log_reg = LogisticRegression(solver='liblinear', max_iter=5000)
log_reg.fit(x_train,y_train)

print('\nCalculated coeficients are:') #m's
print(log_reg.coef_)
print('\nCalculated intercept:') #b
print(log_reg.intercept_)

#%%
#Make predictions
y_pred = log_reg.predict(x_test)

#Print confussion matrix to know how accurate is my project

#True possitive(Upper-left)
#True negative(lower right)
#False positive(top-right)
#False negative(lower-left)
print("\nConfusion matrix results for framework test model 20%")
print(confusion_matrix(y_test,y_pred))
print("Accuracy score for Logistic Regresion with framework is %f" %(accuracy_score(y_test, y_pred)))

#%%
#Logistic regression by hand
from LogisticReg_by_hand import LogisticRegression

X = df_x
y = df_y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

regressor = LogisticRegression(lr=.0001, n_iters=5000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("\nConfusion matrix results for hand test model 20%")
print(confusion_matrix(y_test,y_pred))
print("LR by hand accuracy:", accuracy(y_test, predictions))

#%%
#Test
print("\nAnswer the next questions: \n")
ans1= float(input("Maximum vocal fundamental frequency (Average 197.1049179): "))
ans2= float(input("Minimum vocal fundamental frequency (Average 116.3246308): "))
ans3= float(input("Dysphonic Voice Pattern percetage (Average 0.006220462: "))
ans4= float(input("Absolute Dysphonic Voice Pattern (Average 4.3959E-05: "))
ans5= float(input("Relative Amplitude Perturbation (Average 0.00330641): "))	
ans6= float(input("Point Period Perturbation (Average 0.003446359): "))
ans7= float(input("Shimmer Perturbation Quotient (Average 0.024081487): "))
ans8= float(input("Noice to Armonics Ratio (Average 0.024847077): "))
ans9= float(input("Detrended Fluctuation Analysis (Average 0.718099046): "))
ans10= float(input("Fundamental Frequency Variation (Average 0.226510349): "))				
prediction = log_reg.predict([[ans1,ans2,ans3,ans4,ans5,ans6,ans7,ans8,ans9,ans10]])
print("Results will be printed with 1 as Parkinson case and 0 as healthy")
print("Framework prediction: ")
print(prediction)
prediction2 = regressor.predict([[ans1,ans2,ans3,ans4,ans5,ans6,ans7,ans8,ans9,ans10]])
print("Hand Prediction: ")
print(prediction2)