#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd


# Step 1: Load the data
df = pd.read_csv('Book1.csv')
print("Original Data:")
print(df.head(10))
print("nInfo:")
print(df.info())

# Step 2: Detect missing values
print("nMissing values per column:")
print(df.isnull().sum())

# Step 3: Handle missing values
# Fill missing numerical values with mean
df['quantity'].fillna(df['quantity'].mean(), inplace=True)
df['price'].fillna(df['price'].mean(), inplace=True)

# Fill missing categorical values with 'Unknown'
df['customer_name'].fillna('Uknown', inplace=True)
df['region'].fillna('Unknown', inplace=True)

# Fill missing dates with mode
df['sale_date'].fillna(df['sale_date'].mode()[0], inplace=True)

print("nData after filling missing values:")
print(df.head(10))

# Step 4: Remove duplicates
print("nNumber of duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Step 5: Rename columns to lowercase and replace spaces with underscores
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
print("nColumns after renaming:")
print(df.columns)

# Step 6: Summarize the cleaned data
print("nSummary statistics:")
print(df.describe())

print("nTotal quantity and average price per region:")
region_summary = df.groupby('region').agg({'quantity':'sum', 'price':'mean'})
print(region_summary)

print("nTop 3 products by quantity sold:")
product_summary = df.groupby('product')['quantity'].sum().sort_values(ascending=False).head(3)
print(product_summary)

# Bonus: Create total_sale column
df['total_sale'] = df['quantity'] * df['price']

print("nDate with highest total sales:")
date_sales = df.groupby('sale_date')['total_sale'].sum()
print(date_sales.sort_values(ascending=False).head(1))




# In[ ]:




