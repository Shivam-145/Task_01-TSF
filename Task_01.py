#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION 
# # Task 1:

# Predict the percentage of a student based on the no. of study hours using linear regression
# 

# # SHIVAM KUMAR

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wg
wg.filterwarnings("ignore")


# In[2]:


raw_data='http://bit.ly/w-data'
df=pd.read_csv(raw_data)
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours studied')
plt.ylabel('Scores in percentage')
plt.title('Hours vs percentage')
plt.show()


# In[6]:


# we are checking that there are outliers in the data or not
hours=df['Hours']
scores=df['Scores']


# In[22]:


sns.distplot(hours)


# In[23]:


sns.distplot(scores)


# In[7]:


#dividing the data into training(80%) and testing data(20%)
#linear regression
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)


# In[13]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[14]:


m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x, y)
plt.plot(x, line)
plt.show()


# In[15]:


y_predicted=reg.predict(x_test)
actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_predicted})
print(actual_predicted)


# In[16]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_predicted))
plt.show()


# In[19]:


# what will be the predicted score if a student studies for 9.25 hours/day
h=9.25
p=reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} % in exam ".format(h,p))


# # Model evaluation
# 

# In[21]:


from sklearn import metrics
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_predicted))


# In[ ]:




