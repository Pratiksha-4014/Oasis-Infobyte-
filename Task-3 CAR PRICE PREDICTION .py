#!/usr/bin/env python
# coding: utf-8

# **Name-** Kore Pratiksha Jayant

# **Oasis Infobyte (Data Science)**

# **Task-5** CAR PRICE PREDICTION WITH MACHINE LEARNING

# # 

# **INTRODUCTION-** This project focuses on creating a machine learning model for predicting car prices. We'll analyze factors like brand reputation, car features, horsepower, and mileage to develop an effective prediction system.

# # 

# # ***Importing necessary libraries***

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


# # 

# # ***Loding Dataset***

# In[3]:


data = pd.read_csv('C:\\Users\\stati\\OneDrive\\Desktop\\Car Dataset.csv')


# # 

# # ***EDA (Exploratory Data Analysis)***

# In[4]:


# To check first few rows of the dataframe
(data.head())  


# In[5]:


# To get information about the dataframe
(data.info()) 


# In[6]:


# Displaying a random sample of 5 rows
data.sample(5)


# In[7]:


colunas = data['Car_Name'].str.split(' ', n=1, expand=True)
data['Names'] = colunas[0]


# In[8]:


car_df = data.drop('Car_Name',axis=1)
car_df = data[data['Owner'] !=3]


# In[9]:


data.duplicated().sum()


# In[10]:


# Remove duplicat
new_df = data.drop_duplicates()


# In[11]:


new_df.shape


# In[12]:


new_df.columns


# In[13]:


# Check the number of unique values of each column
new_df.nunique()


# In[14]:


# Checking the distribution of categorical data
categorical_columns = ['Fuel_Type', 'Selling_type', 'Transmission', 'Year', 'Present_Price', 'Owner']

for column in categorical_columns:
    print(new_df[column].value_counts())


# In[15]:


# Descriptive statistics
new_df.describe()


# In[16]:


# List of categorical columns
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']


# In[17]:


numerical_features = ['Year', 'Selling_Price', 'Present_Price', 'Driven_kms', 'Owner']


# In[18]:


# Display unique categories for each categorical column
for column in categorical_features:
    unique_categories = new_df[column].unique()
    print(f"{column} categories: {unique_categories}")
    


# # 

# # ***Visualization***

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# **Create a heatmap of the correlation matrix**

# In[20]:


correlation_matrix = new_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='cubehelix')
plt.title('Correlation Heatmap')
plt.show()


# **Pair Plot**

# In[21]:


sns.pairplot(new_df, hue='Fuel_Type', palette="Set2", height=3)
plt.show()


# In[22]:


sns.pairplot(new_df, hue='Selling_type', palette="Set2", height=3)
plt.show()


# In[23]:


sns.pairplot(new_df, hue='Transmission', palette="Set2", height=3)
plt.show()


# **Bar Plot**

# In[24]:


def plot_bar(x, y, title, size=(12, 6)):
    sns.barplot(x=x, y=y, data=new_df, palette="viridis").set(title=title, xlabel=x, ylabel=y)
    plt.xticks(rotation=90)
    plt.gcf().set_size_inches(size)
    plt.show()


# In[25]:


plot_bar('Names', 'Selling_Price', 'Sales Value by Car Name')


# In[26]:


plot_bar('Names', 'Present_Price', 'Current Sales Amount by Cars Name')


# In[27]:


def plot_count(x, hue, title):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=x, hue=hue, data=new_df, palette='icefire')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.xlabel(x)
    plt.ylabel('Quantity of Cars')
    plt.legend(title=hue)
    plt.show()


# In[28]:


plot_count('Names','Fuel_Type','Quantity of Cars')


# In[29]:


plot_count('Names','Transmission','Quantity of Cars')


# In[30]:


plot_count('Names','Selling_type','Quantity of Cars')


# In[31]:


def plot_bar(x, y, title, size=(10, 6), palette='magma'):
    sns.barplot(x=x, y=y, data=new_df, palette=palette).set(title=title, xlabel=x, ylabel=y)
    plt.gcf().set_size_inches(size)
    plt.show()


# In[32]:


plot_bar('Year', 'Selling_Price', 'Sales Value per Year')


# In[33]:


plot_bar('Year', 'Present_Price', 'Current Sales Value per Year')


# In[34]:


plot_bar('Year', 'Driven_kms', 'Driven Kms per Year')


# In[56]:


**Scatter Plot**


# In[35]:


def plot_scatter(x, y, title, size=4
                 , color='green'):
    plt.figure(figsize=(size, size))
    sns.scatterplot(x=x, y=y, 
                    data=new_df, color=color).set(
        title=title, xlabel=x, ylabel=y)
    plt.show()


# In[36]:


# Rewritten scatter plots with specified size and palette
plot_scatter('Present_Price', 'Selling_Price', 'Sales Value vs Current Value')


# In[37]:


plot_scatter('Selling_Price', 'Driven_kms', 'Sales Value per Driven kms')


# In[38]:


plot_scatter('Present_Price', 'Driven_kms', 'Current Sales Value per Driven kms')


# **Box Plot**

# In[39]:


plt.figure(figsize=(20, 10))

for i, col in enumerate(['Selling_Price', 'Present_Price', 'Driven_kms']):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=data, y='Fuel_Type', x=col, orient='h', color='skyblue')

plt.tight_layout()
plt.show()


# In[40]:


plt.figure(figsize=(20, 10))

for i, col in enumerate(['Selling_Price', 'Present_Price', 'Driven_kms']):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=data, y='Owner', x=col, orient='h', color='skyblue')

plt.tight_layout()
plt.show()


# In[41]:


plt.figure(figsize=(20, 10))

for i, col in enumerate(['Selling_Price', 'Present_Price', 'Driven_kms']):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=data, y='Selling_type', x=col, orient='h', color='skyblue')

plt.tight_layout()
plt.show()


# In[42]:


plt.figure(figsize=(20, 10))

for i, col in enumerate(['Selling_Price', 'Present_Price', 'Driven_kms']):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=data, y='Transmission', x=col, orient='h', color='skyblue')

plt.tight_layout()
plt.show()


# In[43]:


def remove_outliers(col):
    q25, q75 = new_df[col].quantile([0.25, 0.75])
    iqr = q75 - q25
    upper_limit, lower_limit = q75 + 1.5 * iqr, q25 - 1.5 * iqr
    return new_df[(new_df[col] >= lower_limit) & (new_df[col] <= upper_limit)]

# Remove outliers for each specified column
data = remove_outliers('Selling_Price')
data = remove_outliers('Present_Price')
data = remove_outliers('Driven_kms')


# In[44]:


from sklearn.preprocessing import LabelEncoder

# Define categorical columns for encoding
categorical_columns = ['Fuel_Type', 'Selling_type', 'Transmission', 'Car_Name']

# Create dictionaries to store label mappings
label_mapping = {}

# Apply label encoding and store mappings
for col in categorical_columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    label_mapping[col] = dict(enumerate(encoder.classes_))

data.head()


# In[45]:


X = data.drop('Selling_Price', axis=1).values
Y = data['Selling_Price'].values


# In[46]:


print(X.shape)
print(type(X))


# In[47]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Categorical column indices
categorical_cols = [5, 6, 7, 8]  # Adjust these indices based on your data

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(categories='auto', sparse=False, drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit and transform the data
X_encoded = preprocessor.fit_transform(X)


# In[48]:


#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)


# In[49]:


#Train a Regression Model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# In[50]:


y_pred_linear = linear_model.predict(X_test)


# In[51]:


#Evaluating the Regression Model
from sklearn.metrics import mean_squared_error
from math import sqrt
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = sqrt(mse_linear)
print(f'Linear Regression RMSE: {rmse_linear}')


# In[52]:


#Train a Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)


# In[53]:


y_pred_rf = rf_model.predict(X_test)


# In[54]:


#Evaluating the Random Forest Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = sqrt(mse_rf)
print(f'Random Forest RMSE: {rmse_rf}')


# In[55]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price (Random Forest)')
plt.title('Actual vs. Predicted Selling Price (Random Forest)')
plt.show()


# As wev can see from above scatter plot that datapoints are close to eachother we can say that our model works well.
