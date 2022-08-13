#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
sns.set()


# In[2]:


df = pd.read_csv("titanic.csv")


# In[3]:


df.shape


# In[4]:


df.info()


# ## Data Description
# 
# - Survival : 0 = No, 1 = Yes
# - pclass(Ticket Class) : 1 = 1st, 2 = 2nd, 3 = 3rd
# - Sex(Gender) : Male, Female
# - Age : Age in years
# - SibSp : Number of siblings/spouses abroad the titanic
# - Parch : Number of parents/children abrod the titanic
# - Ticket : Ticket Number
# - Fare : Passenger fare
# - Cabin : Cabin Number
# - Embarked : Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Southampton

# # Q1: Print first and last rows from the dataset

# In[5]:


df.head()


# In[6]:


df.tail()


# # Q2: Select only survived people

# In[7]:


df.loc[0:890,[ 'survived']]


# # Q3: Select sex, fare, survived columns

# In[8]:


df.loc[0:890,['sex', 'fare', 'survived']]


# # Q4: Add a new_column to a DataFrame that combines class and embark_town

# In[9]:


df['new_column'] = df['class'].astype(str) + '-' + df['embark_town']
df


# # Q5: Remove new_column from the DataFrame

# In[14]:


df.drop(['new_column'], axis = 1)


# # Q6: Filter DataFrame for rows of survived Males only 

# In[15]:


filt = (df['survived'] == 'male')


# In[16]:


df[filt]


# # Q7: The total number of males who survived 

# In[17]:


df.info('male')


# # Q8: How many values in each class?

# In[18]:


class_value = df['class'].tolist()
class_value


# import seaborn as sns

# In[ ]:





# # Q9: Draw barplot represents survived people based on sex

# In[25]:


sns.barplot(x='survived',y='sex', data=df);


# # Q10: Draw catplot represents survived people based on embarked

# In[29]:


g = sns.catplot(
    data=df, 
    kind="bar", 
    x="survived",
    y="embarked",
)
plt.title("survived people based on embarked");


# # Q11: Draw boxplot represents distribution of male and female based on age and pclass

# In[50]:


plt.figure(figsize=(15,8))

sns.boxplot(
    x="age", 
    y="pclass",
    hue="age",
    data=df)

plt.title("distribution of male and female based on age and pclass");


# # Q12: Draw heatmap represents correlations between sibsp, parch, age, fare, and survived columns

# In[37]:


corr=df[["sibsp","parch","fare","survived",]].corr()
sns.heatmap(corr,annot=True);


# # Q13: Draw factorplot represents the relation between sibsp and survived columns

# In[38]:


sns.factorplot(x='sibsp',y='survived',data=df);


# # Q14: Draw extra insights [Optional]

# In[ ]:


df.head
len

