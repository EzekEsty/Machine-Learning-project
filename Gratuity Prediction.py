#!/usr/bin/env python
# coding: utf-8

# In[1]:


x = 10
type(x)


# In[2]:


x = str(x)
type(x)


# In[3]:


greeting = 'hello '
output = greeting*5
print(output)


# In[4]:


course = 'python'
print(len(course))
upper = course.upper()
print(upper)


# In[5]:


greeting1 = 'Hello'
greeting2 = 'students!'
print(greeting1 + ' ' + greeting2)


# In[6]:


#Slicing. i.e extracting values or characters

greeting = 'Hello'
print(greeting[0])


# In[7]:


print(greeting[0:4])


# In[8]:


greeting = 'cardinal'
print(greeting[-3:])


# In[9]:


a = 10
if a > 5:
    print("The value is greater than 5")


# In[10]:


a = 10
if a > 15:
    print("The value is greater than 15")


# In[11]:


a = 4
if a > 10:
    print("The value is greater than 10")
else:
    print("The value is less than 10")


# # Working with Pandas

# In[12]:


import pandas as pd


# In[13]:


#list

data = [['Adebayo', 20], ['Esther', 15], ['Ezekiel', 12], ['Boluwatife', 10], ['Chiwendu', 15]]

#Turn the above data to dataframe and name the columns
df = pd.DataFrame(data, columns = ['Name', 'Age'])
df


# Using Dictionaries of narray list

# In[14]:


D = {'Name': ['Adebayo', 'Ezekiel', 'Esther', 'Boluwatife', 'Chiwendu'], 'Age': [12, 13, 14, 15, 16]}
DF = pd.DataFrame(D)
DF


# In[15]:


#Creating state, capital, area and polpulation

pop = {'State': ['Abia', 'Adamawa', 'Imo', 'Kwara'], 'Capital': ['Umuahia', 'Yola', 'Owerri', 'Ilorin'], 'Area': [1234, 2345, 3421, 2134], 'Population': [1234567, 2201230, 34521092, 12310022]}

Data = pd.DataFrame(pop)
Data


# IMPORTING DATA

# In[16]:


K = pd.read_excel("C:/Users/User/Documents/AI DATA SET/Dataset AI Invasion/2006.xlsx")
K.head(5)


# In[17]:


K.describe()


# In[18]:


#To include the non numerical variables
K.describe(include = 'all')


# In[19]:


#To get the columns
K.columns


# In[20]:


#To get more information
K.info()


# In[21]:


#To get the state column, population column and both together
print((K['STATES']).head(10))


# In[22]:


print((K['Population']).head(10))


# In[23]:


print((K[['STATES', 'Population']]).head(10))


# In[24]:


#To print out the first four observations
print(K[0:4])


# In[25]:


#To print the 5th, 6th and 7th observations
print(K[5:8])


# In[26]:


#Using iloc to get objects
print(K.iloc[3])


# Dropping Features and Rows in a Dataset

# In[27]:


#To drop the population column/feature
print(K.drop(['Population'], axis = 1))


# In[28]:


#To add features to the dataset
K['Population'] = K['Population']
K


# Changing Data type of Pandas datafram and pandas series

# In[29]:


#To change the data type of population to float
K['Population'] = K['Population'].astype('float')
K.head(10)


# Pandas Sorting Method: index or value

# In[30]:


K.sort_index(inplace = True, ascending = False)


# In[31]:


K


# In[32]:


#Sorting with Value

K.sort_values('STATES', axis = 0, ascending = False, inplace = True)
K


# The objective of the regression task is to predict the amount of tip (gratuity in Nigeria naira) 
# given to a food server based on total_bill, gender, smoker (whether they smoke in the party or 
# not), day (day of the week for the party), time (time of the day whether for lunch or dinner), 
# and size (size of the party) in Mama Tee restaurant..

# In[33]:


#Predicting tip using the method od OLS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


Tip = pd.read_csv("C:/Users/User/Documents/AI DATA SET/Dataset AI Invasion/tips.csv")
Tip.head(10)


# In[35]:


Tip.shape


# Relationship with categorical Variables

# In[36]:


#Using boxplot to find the relationship between tip vs gender

sns.boxplot(x = "gender", y = "tip", data = Tip)
plt.ylabel("Amount of Tip")


# From the above chart, there is no much difference in the amount of tip given by both genders. Also, there was an extreme amount of tip given to the male food servers.

# In[37]:


#Relationship between tip and smokers

sns.boxplot(x = "smoker", y = "tip", data = Tip)
plt.ylabel("Amount of tip")


# From the above plot, both smokers and non-smokers gave almost the same amount of tip

# In[38]:


#Relationship between tip and time

sns.boxplot(x = "time", y = "tip", data = Tip)
plt.ylabel("Amount of tip")


# Those who took lunch and dinner gave almost the same amount of tip. Though an extreme amount of tip was given by those who took dinner.

# # Model building
# 
# After getting some insight about the data, the data is now prepared for machine learning 
# modelling

# In[39]:


#For model evaluation
from sklearn import metrics

#To divide the data into training and test set
from sklearn.model_selection import train_test_split


# # Data Preprocessing
# ● Separating features and the label from the data

# In[40]:


#To split the data into features(X) and targets(Y)

#To drop the tip variable-label from the data and name it x
x = Tip.drop(["tip"], axis = "columns")
y = Tip["tip"]


# In[41]:


x.head(5)


# In[42]:


y.head(5)


# # Since the label is continuous, this is a regression task.
# 
# ● One-hot encoding
# 
# We need to create a one-hot encoding for all the categorical features in 
# the data because some algorithms cannot work with categorical data directly. They require all 
# input variables and output variables to be numeric. In this case, we will create a one-hot 
# encoding for gender, smoker, day and time by using pd.get_dummies()

# In[43]:


X = pd.get_dummies(x)
X


# In[44]:


X.shape


# # ● Split the data into training and test set
# 
# The dataset splitted into (Features (X) and Label (Y)) into training and test data by using 
# train_test_split() function from sklearn. The training set will be 80% while the test set will be 
# 20%. The random_state that is set to 1234.

# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


# Since the data is splitted, we now have (X_train, y_train) and (X_test, y_test) 

# # ● Training the Model
# The training data is used to build the model and then use test data to make prediction and 
# evaluation respectively

# Linear Regression.
# Since we are to train a linear regression model with our training data. We need to import the Linear 
# regression from the sklearn model.

# In[46]:


#To fit linear Regression to our training set

from sklearn.linear_model import LinearRegression


# In[47]:


linearmodel = LinearRegression()
linearmodel.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)


# Since the linearmodel.fit has trained the Linear regression model.
# The model is now ready to make predictions for the unknown label by using only the features from the test data (X_test)

# In[48]:


linearmodel.predict(X_test)


# We can save the above prediction result with linearmodel_prediction. The above result is what the model predicted.

# In[49]:


linearmodel_prediction = linearmodel.predict(X_test)


# Since the prediction is continuous, we can only measure how far the prediction is from the 
# actual values. Let's check the error for each prediction. The error prediction can be calculated by subtracting the predicted frrom the actual

# In[50]:


y_test - linearmodel_prediction


# From the above results, it is shown that the positive ones show that the prediction is higher than the actual values while the negative 
# ones are below the actual values.

# We want to measure this error by using the Root Mean Squared Error (RMSE).

# In[51]:


#The MSE between the actual and the predicted
MSE = metrics.mean_squared_error(y_test, linearmodel_prediction)


# In[52]:


MSE


# In[53]:


#To get the root MSE
np.sqrt(MSE)


# Therefore, the RMSE for the linear regression is 142.1316828752442.

# We have to train with other models too to check for the model

# # TO Use The Random Forest Regressor

# In[54]:


#First, import the model from the sklearn module

from sklearn.ensemble import RandomForestRegressor
randomforestmodel = RandomForestRegressor()


# In[55]:


randomforestmodel.fit(X_train, y_train)


# In[57]:


RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',max_depth=None,max_features='auto',max_leaf_nodes=None,max_samples=None,min_impurity_decrease=0.0,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_jobs=None,oob_score=False,random_state=None,verbose=0,warm_start=False)


# Random Forest model has been used to train the data. Now we can make predictions

# In[58]:


randomforestmodel_prediction = randomforestmodel.predict(X_test)


# In[59]:


randomforestmodel_prediction


# In[60]:


#To get the error between the actual and the predicted
MSE = metrics.mean_squared_error(y_test, randomforestmodel_prediction)
MSE


# In[61]:


#To get the root mean squared error
np.sqrt(MSE)


# Therefore, the RMSE for the Random Forest Model is 159.54570529200726

# In[62]:


#To check the difference between the actual and predicted tip
y_test - randomforestmodel_prediction


# # To use the SVM (Support Vector Machine) model

# In[66]:


from sklearn.svm import SVR
SVMmodel = SVR()


# In[67]:


SVMmodel.fit(X_train, y_train)


# In[70]:


SVR(C = 1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)


# In[71]:


SVMmodel_prediction = SVMmodel.predict(X_test)
SVMmodel_prediction


# In[72]:


MSE = metrics.mean_squared_error(y_test, SVMmodel_prediction)
MSE


# In[73]:


#To get the root mean square
np.sqrt(MSE)


# Therefore, the RMSE for the Support Vector Machine (SVM) is 140.90188181480886

# # To use the Decision Tree Model

# In[74]:


#Import the decisin tree model
from sklearn.tree import DecisionTreeRegressor
decisiontree = DecisionTreeRegressor()
decisiontree.fit(X_train, y_train)


# In[78]:


DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0, random_state=None, splitter='best')


# In[81]:


decisiontree_prediction = decisiontree.predict(X_test)
decisiontree_prediction


# In[82]:


MSE = metrics.mean_squared_error(y_test, decisiontree_prediction)
MSE


# In[83]:


np.sqrt(MSE)


# In[84]:


#To get the differece between the actual tip and predicted tip
y_test - decisiontree_prediction


# Therefore, the RMSE for the Decision Tree is 215.84571333501313.

# Having trained all the four (4) models, we can see that the best model that can accurately 
# predict the amount of tips that would be given for a given party in the restaurant is the model 
# with the lowest RMSE and that is Support Vector Machine (SVM).

# # To get the amount of Tip using the next three (3) models: K-Nearset neighbor, Ridge Regression and Gradient Boost

# In[85]:


#Using the K-Nearest Neighbor

from sklearn.neighbors import KNeighborsRegressor
KNeighbor = KNeighborsRegressor()


# In[86]:


KNeighbor.fit(X_train, y_train)


# In[87]:


KNeighbor_prediction = KNeighbor.predict(X_test)
KNeighbor_prediction


# In[89]:


#To get the MSE
MSE = metrics.mean_squared_error(y_test, KNeighbor_prediction)
MSE


# In[90]:


#To get the root MSE
np.sqrt(MSE)


# Therefore, the RMSE for the K Nearest Neighbors is 161.09883126634577

# # Using the Ridge Regression model

# In[91]:


from sklearn.linear_model import Ridge

ridge = Ridge()


# In[92]:


ridge.fit(X_train, y_train)


# In[93]:


ridge_prediction = ridge.predict(X_test)
#To check the predicted values
ridge_prediction


# In[94]:


MSE = metrics.mean_squared_error(y_test, ridge_prediction)
MSE


# In[95]:


np.sqrt(MSE)


# Therefore, the RMSE for the Ridge Regression is 142.13005118998743

# In[97]:


#To check the difference between the actual and predicted Tip
y_test - ridge_prediction


# # Using the Gradient Boost Model

# In[98]:


from sklearn.ensemble import GradientBoostingRegressor

Gradient = GradientBoostingRegressor()


# In[99]:


Gradient.fit(X_train, y_train)


# In[100]:


Gradient_prediction = Gradient.predict(X_test)
Gradient_prediction


# In[101]:


MSE = metrics.mean_squared_error(y_test, Gradient_prediction)
MSE


# In[102]:


np.sqrt(MSE)


# Therefore, the RMSE for the Gradient Boost model is 160.7848172103318

# In[103]:


#To check the difference between the actual tip and the predicted tip
y_test - Gradient_prediction


# Having trained all the three (3) models, we can see that the best model that can accurately predict the amount of tips that would be given for a given party in the restaurant is the model with the lowest RMSE and that is Ridge Regression model = 142.13005118998743).

# In[ ]:




