#!/usr/bin/env python
# coding: utf-8

# # EDA Assignement

# In[1]:


###import libraries to read files


# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
from plotly import tools
import plotly.express as px
import pandas.core.algorithms as algos
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[3]:


Application = pd.read_csv('application_data.csv')
PreviousApplication = pd.read_csv('previous_application.csv')


# # 1. Application - Data Routine Check

# In[4]:


Application.info()


# In[5]:


pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200) 
pd.set_option('display.width', 1000)


# In[6]:


Application.dtypes


# # 2. Clean Data 

# In[7]:


Application.isnull().mean().sort_values(ascending = True)


# In[8]:


unwanted=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

Application.drop(labels=unwanted,axis=1,inplace=True)


# # 2.2 Drop columns having 40% missing 
# 

# In[9]:


Application_null = Application.isnull().sum()
Application_null = Application_null[Application_null.values > (0.40 * len(Application))]
len(Application_null)


# In[10]:


Application_null=list(Application_null.index)
Application.drop(labels=Application_null, axis=1, inplace=True)


# In[11]:


Application.shape


# # Application DATA

# # 2.3 Convert negative values to positive 

# In[12]:



filter_days = [col for col in Application if col.startswith('DAYS')]
print(Application[filter_days])


# In[13]:


Application[filter_days] = abs(Application[filter_days])
print(Application[filter_days])


# In[ ]:





# # 2.4 Imputation/Drop missing Values

# In[14]:


Application.CODE_GENDER.value_counts()


# In[15]:


# Replacing XNA with NaN
Application = Application.replace('XNA',np.NaN)


# In[16]:


Application.drop(Application[(Application['CODE_GENDER'] == "XNA")].index, inplace=True)


# In[17]:


Application.CODE_GENDER.value_counts()


# In[18]:


list(Application.columns[(Application.isnull().mean()<=0.40) & (Application.isnull().mean()>0)])


# In[19]:


#### Impute AMT_ANNUITY


# In[20]:


print(Application['AMT_ANNUITY'].value_counts())
AMT_ANNUITY_mean= Application['AMT_ANNUITY'].mean()
AMT_ANNUITY_median= Application['AMT_ANNUITY'].median()
print('mean =',AMT_ANNUITY_mean)
print('median =',AMT_ANNUITY_median)
sns.boxplot(Application['AMT_ANNUITY'])
plt.show()


# In[21]:


Application['AMT_ANNUITY'].fillna(AMT_ANNUITY_median, inplace=True) ##Replace missing value with median


# In[22]:


### Impute AMT_GOODS_PRICE


# In[23]:


print(Application['AMT_GOODS_PRICE'].value_counts())
AMT_GOODS_PRICE_mean= Application['AMT_GOODS_PRICE'].mean()
AMT_GOODS_PRICE_median= Application['AMT_GOODS_PRICE'].median()
print('mean =',AMT_GOODS_PRICE_mean)
print('median =',AMT_GOODS_PRICE_median)
sns.boxplot(Application['AMT_GOODS_PRICE'])
plt.show()


# In[24]:


Application['AMT_GOODS_PRICE'].fillna(AMT_GOODS_PRICE_median, inplace=True) ##Replace missing value with median


# In[25]:


print(Application['NAME_TYPE_SUITE'].value_counts())


# In[26]:


Application['NAME_TYPE_SUITE'].fillna("Unaccompanied", inplace=True) ##Replace missing value with Most common value


# In[27]:


print(Application['OCCUPATION_TYPE'].value_counts())


# In[28]:


Application['OCCUPATION_TYPE'].fillna("Laborers", inplace=True) ##Replace missing value with Most common value


# In[29]:


print(Application['CNT_FAM_MEMBERS'].value_counts())
CNT_FAM_MEMBERS_mean= Application['CNT_FAM_MEMBERS'].mean()
CNT_FAM_MEMBERS_median= Application['CNT_FAM_MEMBERS'].median()
print('mean =',CNT_FAM_MEMBERS_mean)
print('median =',CNT_FAM_MEMBERS_median)
sns.boxplot(Application['CNT_FAM_MEMBERS'])
plt.show()


# In[30]:


Application['CNT_FAM_MEMBERS'].fillna(CNT_FAM_MEMBERS_median, inplace=True) ##Replace missing value with median


# In[31]:


print(Application['EXT_SOURCE_2'].value_counts())
EXT_SOURCE_2_mean= round(Application['EXT_SOURCE_2'].mean(),6)
EXT_SOURCE_2_median=round(Application['EXT_SOURCE_2'].median(),6)
print('mean =',EXT_SOURCE_2_mean)
print('median =',EXT_SOURCE_2_median)
sns.boxplot(Application['EXT_SOURCE_2'])
plt.show()


# In[32]:


Application['EXT_SOURCE_2'].fillna(EXT_SOURCE_2_median, inplace=True) ##Replace missing value with median


# In[33]:


print(Application['EXT_SOURCE_3'].value_counts())
EXT_SOURCE_3_mean= round(Application['EXT_SOURCE_3'].mean(),6)
EXT_SOURCE_3_median=round(Application['EXT_SOURCE_3'].median(),6)
print('mean =',EXT_SOURCE_3_mean)
print('median =',EXT_SOURCE_3_median)
sns.boxplot(Application['EXT_SOURCE_3'])
plt.show()


# In[34]:


Application['EXT_SOURCE_3'].fillna(EXT_SOURCE_3_median, inplace=True) ##Replace missing value with median


# In[35]:


print(Application['OBS_30_CNT_SOCIAL_CIRCLE'].value_counts())
OBS_30_CNT_SOCIAL_CIRCLE_mean= round(Application['OBS_30_CNT_SOCIAL_CIRCLE'].mean(),6)
OBS_30_CNT_SOCIAL_CIRCLE_median=round(Application['OBS_30_CNT_SOCIAL_CIRCLE'].median(),6)
print('mean =',OBS_30_CNT_SOCIAL_CIRCLE_mean)
print('median =',OBS_30_CNT_SOCIAL_CIRCLE_median)
sns.boxplot(Application['OBS_30_CNT_SOCIAL_CIRCLE'])
plt.show()


# In[36]:


Application['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(OBS_30_CNT_SOCIAL_CIRCLE_median, inplace=True) ##Replace missing value with median because privous all missing values are rplace with median & 0.0 pcuur maximum time


# In[37]:


print(Application['DEF_30_CNT_SOCIAL_CIRCLE'].value_counts())


# In[38]:


Application['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[39]:


print(Application['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts())


# In[40]:


Application['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[41]:


print(Application['DEF_60_CNT_SOCIAL_CIRCLE'].value_counts())


# In[42]:


Application['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[43]:


print(Application['AMT_REQ_CREDIT_BUREAU_HOUR'].value_counts())


# In[44]:


Application['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[45]:


print(Application['AMT_REQ_CREDIT_BUREAU_DAY'].value_counts())


# In[46]:


Application['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[47]:


print(Application['AMT_REQ_CREDIT_BUREAU_WEEK'].value_counts())


# In[48]:


Application['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[49]:


print(Application['AMT_REQ_CREDIT_BUREAU_MON'].value_counts())


# In[50]:


Application['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[51]:


print(Application['AMT_REQ_CREDIT_BUREAU_QRT'].value_counts())


# In[52]:


Application['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0.0, inplace=True) ##Replace missing value with common value


# In[53]:


print(Application['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts())
AMT_REQ_CREDIT_BUREAU_YEAR_mean= round(Application['AMT_REQ_CREDIT_BUREAU_YEAR'].mean(),6)
AMT_REQ_CREDIT_BUREAU_YEAR_median=round(Application['AMT_REQ_CREDIT_BUREAU_YEAR'].median(),6)
print('mean =',AMT_REQ_CREDIT_BUREAU_YEAR_mean)
print('median =',AMT_REQ_CREDIT_BUREAU_YEAR_median)
sns.boxplot(Application['AMT_REQ_CREDIT_BUREAU_YEAR'])
plt.show()


# In[54]:


Application['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(AMT_REQ_CREDIT_BUREAU_YEAR_median, inplace=True) ##Replace missing value with median


# In[ ]:





# # 3. Data Analysis

# # 3.1 Find Outliers & remove

# In[55]:


# boxplot for 'AMT_INCOME_TOTAL' column to check the outliers
fig = px.box(Application, y="AMT_INCOME_TOTAL",title='AMT_INCOME_TOTAL' )
fig.show()


# In[56]:


#### 127M Income deifinetly outlier need to drop this row
Application.drop(Application[(Application['AMT_INCOME_TOTAL'] > 116000000) ].index, inplace=True)


# In[57]:


fig = px.box(Application, y="AMT_INCOME_TOTAL",title='AMT_INCOME_TOTAL' )
fig.show()


# In[58]:


# boxplot for 'DAYS_BIRTH' column to check the outliers
fig = px.box(Application, y="DAYS_BIRTH",title='DAYS_BIRTH' )
fig.show()


# In[59]:


####NO ouliers in AMT_ANNUITY


# In[60]:


fig = px.box(Application, y="AMT_ANNUITY",title='AMT_ANNUITY' )
fig.show()


# In[61]:


##### 258K is outlier


# In[62]:


#### 127M Income deifinetly outlier need to drop this row
Application.drop(Application[(Application['AMT_ANNUITY'] > 250000) ].index, inplace=True)


# In[63]:


fig = px.box(Application, y="AMT_ANNUITY",title='AMT_ANNUITY' )
fig.show()


# In[64]:


fig = px.box(Application, y="AMT_CREDIT",title='AMT_CREDIT' )
fig.show()


# In[65]:


##### There are only few AMT credit more than 3.5M so we can consiter it as outliers


# In[66]:


#### >3.5M Income deifinetly outlier need to drop this row
Application.drop(Application[(Application['AMT_CREDIT'] > 3500000) ].index, inplace=True)


# In[67]:


fig = px.box(Application, y="AMT_CREDIT",title='AMT_CREDIT' )
fig.show()


# In[68]:


fig = px.box(Application, y="AMT_GOODS_PRICE",title='AMT_GOODS_PRICE' )
fig.show()


# In[69]:


#### >3M Income deifinetly outlier need to drop this row
Application.drop(Application[(Application['AMT_GOODS_PRICE'] > 3000000) ].index, inplace=True)


# In[70]:


fig = px.box(Application, y="AMT_GOODS_PRICE",title='AMT_GOODS_PRICE' )
fig.show()


# In[71]:


# Converting 'DAYS_EMPLOYED' to years
Application['DAYS_EMPLOYED']= (Application['DAYS_EMPLOYED']/365).astype(int)


# In[72]:


fig = px.box(Application, y="DAYS_EMPLOYED",title='DAYS_EMPLOYED' )
fig.show()


# In[73]:


#### DAYS_EMPLOYED 1000 years not possible its human error so drop this entry


# In[74]:


Application.drop(Application[(Application['DAYS_EMPLOYED'] > 100) ].index, inplace=True)


# In[75]:


fig = px.box(Application, y="DAYS_EMPLOYED",title='DAYS_EMPLOYED' )
fig.show()


# In[76]:


fig = px.box(Application, y="DAYS_REGISTRATION",title='DAYS_REGISTRATION' )
fig.show()


# In[77]:


#### No outliers in DAYS_REGISTRATION


# In[78]:


# Dividing the original dataset into two different datasets depending upon the target value
target0 = Application.loc[Application.TARGET == 0]
target1 = Application.loc[Application.TARGET == 1]


# In[79]:


DataImbalanceRatio= len(target0)/len(target1)
round(DataImbalanceRatio,2)


# In[80]:


####Data is highly imbalance 


# # Application Data Bining

# In[81]:


# AMT_INCOME_TOTAL can be binned into new column AMT_INCOME_RANGE
Application['AMT_INCOME_RANGE'] = pd.qcut(Application.AMT_INCOME_TOTAL, q=[0, 0.2, 0.5, 0.8, 0.9, 1], labels=['VERY_LOW', 'LOW', "MEDIUM", 'HIGH', 'VERY_HIGH'])
print(Application['AMT_INCOME_RANGE'].head())


# In[82]:


Application['AMT_CREDIT_RANGE'] = pd.qcut(Application.AMT_CREDIT, q=[0, 0.2, 0.5, 0.8, 0.9, 1], labels=['VERY_LOW', 'LOW', "MEDIUM", 'HIGH', 'VERY_HIGH'])
print(Application['AMT_CREDIT_RANGE'].head())


# In[83]:


# Converting 'DAYS_BIRTH' to years, which will be an approximate age
Application['DAYS_BIRTH']= (Application['DAYS_BIRTH']/365).astype(int)
Application['DAYS_BIRTH'].unique()


# In[84]:


Application['AGE_RANGE']=pd.cut(Application['DAYS_BIRTH'], bins=[19,25,40,60,100], labels=['Teenager','Adult', 'Senior Adult', 'Senior_Citizen'])
Application.AGE_RANGE.value_counts()


# # Application - Data Distribution

# In[85]:


# Distribution of 'OCCUPATION_TYPE'
Dist = Application["OCCUPATION_TYPE"].value_counts()
fig = px.bar(Dist,title='Applicants Occupation' )
fig.show()


# In[86]:


# Distribution of 'ORGANIZATION_TYPE'
Dist = Application["ORGANIZATION_TYPE"].value_counts()
fig = px.bar(Dist,title='Organizations - Applied for Loans')
fig.show()


# #  Dividing dataset into two dataset (Defaulter & Non-Defaulter)

# In[87]:


# target1(client with payment difficulties) & target0(all other)

target0=Application.loc[Application["TARGET"]==0]
target1=Application.loc[Application["TARGET"]==1]


# # Application Data Imbalance check

# In[88]:


data_Imbalance = Application["TARGET"].value_counts().reset_index()
plt.figure(figsize=(12,7))
x= ['Non_Defaulter','Defaulter']
sns.barplot(x,"TARGET",data = data_Imbalance,palette= ['g','b'])
plt.xlabel("Loan payment Status")
plt.ylabel("Count Non_Defaulter & Defaulters")
plt.title("Data Imbalance")
plt.show()
Application['TARGET'].value_counts(normalize=True)*100


# In[89]:


# Univariate Analysis on Application Data


# In[90]:


# function to plot for numerical variables (Target0 and Target1)
def Univariate_bar(var):
    fig = make_subplots(rows=1, cols=2,horizontal_spacing=0.2,
                        subplot_titles=('Distribution of '+ '%s' %var +' for Non-Defaulters','Distribution of '+ '%s' %var +' for Defaulters'))
    fig.add_trace(go.Bar(x=target0[var].unique(), y=target0[var].value_counts()), 1, 1)
    fig.add_trace(go.Bar(x=target1[var].unique(), y=target1[var].value_counts()), 1, 2)
    fig.update_layout(showlegend=False)
    fig.show()


# In[91]:


# function to plot pie chart 
def Univariate_pie(var):
    label_0 = target0[var].values
    label_1 = target1[var].values
    fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=[' Non_Defaulters',' Defaulters'])
    fig.add_trace(go.Pie(labels=label_0), 1, 1)
    fig.add_trace(go.Pie(labels=label_1), 1, 2)
    fig.show()


# In[92]:


#Gender Distribution
Univariate_pie('CODE_GENDER')


# In[93]:


#Age Distibution
Univariate_bar('AGE_RANGE')


# In[94]:


#NAME_INCOME_TYPE
Univariate_bar('NAME_INCOME_TYPE')


# In[95]:


#NAME_FAMILY_STATUS
Univariate_bar('NAME_FAMILY_STATUS')


# In[96]:


#NAME_EDUCATION_TYPE
Univariate_bar('NAME_EDUCATION_TYPE')


# In[97]:


#NAME_TYPE_SUITE
Univariate_bar('NAME_TYPE_SUITE')


# In[98]:


Univariate_pie('NAME_CONTRACT_TYPE')


# In[99]:


Univariate_pie('CNT_FAM_MEMBERS')


# In[100]:


#INCOME_RANGE
Univariate_bar('AMT_INCOME_RANGE')


# # Bivariate Analysis on Application Data

# In[101]:


#Pairplot for Target 0 (Loan-Non Payment Defaulters)
pair = target0[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'DAYS_BIRTH']].fillna(0)
sns.pairplot(pair)

plt.show()


# In[102]:


#Pairplot for Target 1 (Loan-Payment Difficulties)
pair = target1[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'DAYS_BIRTH']].fillna(0)
sns.pairplot(pair)

plt.show()


# In[103]:


# 'NAME_EDUCATION_TYPE' vs 'AMT_CREDIT' for Loan - Non Defaulters
fig = px.box(target0, x="NAME_EDUCATION_TYPE", y="AMT_CREDIT", color='NAME_FAMILY_STATUS',
             title="Credit amount vs Education of Loan - Non Defaulters")
fig.show()


# In[104]:


# 'NAME_EDUCATION_TYPE' vs 'AMT_CREDIT' for Loan Defaulters
fig = px.box(target1, x="NAME_EDUCATION_TYPE", y="AMT_CREDIT",color='NAME_FAMILY_STATUS',
                title="Credit amount vs Education of Loan Defaulters")

fig.show()


# In[105]:


# Between AMT_INCOME_TOTAL vs AGE_RANGE Gender wise


# In[106]:


# 'AMT_INCOME_TOTAL' vs 'AGE_RANGE' for Loan - Defaulters
fig = px.box(target0, x="AMT_INCOME_RANGE", y="AMT_CREDIT",color='FLAG_OWN_REALTY',
             title="Credit amount vs Education of Loan - Non Defaulters")
fig.show()


# In[107]:


# 'AMT_INCOME_TOTAL' vs 'AGE_RANGE' for Loan - Defaulters
fig = px.box(target1, x="AMT_INCOME_RANGE", y="AMT_CREDIT",color='FLAG_OWN_REALTY',
             title="Credit amount vs Education of Loan - Non Defaulters")
fig.show()


# #  Correlation for numerical columns for both target cases

# In[108]:


# Correlation for Loan- Non Payment Defaulters
target0[['AMT_GOODS_PRICE','AMT_INCOME_TOTAL','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_BIRTH',
                            'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','AMT_CREDIT','CNT_CHILDREN']].corr(method = 'pearson').iplot(kind='heatmap',colorscale="RdYlGn",title="Correlation Heatmap of Loan- Non Difaulters")


# In[109]:


# Heatmap for Loan- Payment Defaulters
target1[['AMT_GOODS_PRICE','AMT_INCOME_TOTAL','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_BIRTH',
                            'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','AMT_CREDIT','CNT_CHILDREN']].corr(method = 'pearson').iplot(kind='heatmap',colorscale="RdYlGn",title="Correlation Heatmap of Loan-Difaulters")


# In[ ]:





# In[ ]:





# # Data Analysis For Previous Application Data

# In[110]:


PreviousApplication.info()


# In[111]:


PreviousApplication.head(5)


# In[112]:


PreviousApplication.isnull().mean().sort_values(ascending = True)


# In[113]:


PreviousApplication_null = PreviousApplication.isnull().sum()
PreviousApplication_null = PreviousApplication_null[PreviousApplication_null.values > (0.40 * len(PreviousApplication))]
len(PreviousApplication_null)


# In[114]:


PreviousApplication_null=list(PreviousApplication_null.index)
PreviousApplication.drop(labels=PreviousApplication_null, axis=1, inplace=True)
PreviousApplication.shape


# In[115]:


# Replacing XNA with NaN
PreviousApplication = PreviousApplication.replace('XNA',np.NaN)
# Replacing XAP with NaN
PreviousApplication = PreviousApplication.replace('XAP',np.NaN)


# In[116]:


list(PreviousApplication.columns[(PreviousApplication.isnull().mean()<=0.40) & (PreviousApplication.isnull().mean()>0)])


# In[117]:


#### AMT_ANNUITY


# In[118]:


print(PreviousApplication['AMT_ANNUITY'].value_counts())
AMT_ANNUITY_mean= PreviousApplication['AMT_ANNUITY'].mean()
AMT_ANNUITY_median= PreviousApplication['AMT_ANNUITY'].median()
print('mean =',AMT_ANNUITY_mean)
print('median =',AMT_ANNUITY_median)
sns.boxplot(PreviousApplication['AMT_ANNUITY'])
plt.show()


# In[119]:


PreviousApplication['AMT_ANNUITY'].fillna(AMT_ANNUITY_median, inplace=True) ##Replace missing value with median


# In[120]:


print(PreviousApplication['AMT_CREDIT'].value_counts())
AMT_CREDIT_mean= PreviousApplication['AMT_CREDIT'].mean()
AMT_CREDIT_median= PreviousApplication['AMT_CREDIT'].median()
print('mean =',AMT_CREDIT_mean)
print('median =',AMT_CREDIT_median)
sns.boxplot(PreviousApplication['AMT_CREDIT'])
plt.show()


# In[121]:


### We can take 0.0 because its frequently appear value but because we taking median value maximum time here also we are taking median to rplace missing value


# In[122]:


PreviousApplication['AMT_CREDIT'].fillna(AMT_CREDIT_median, inplace=True) ##Replace missing value with median


# In[123]:


print(PreviousApplication['AMT_GOODS_PRICE'].value_counts())
AMT_GOODS_PRICE_mean= PreviousApplication['AMT_GOODS_PRICE'].mean()
AMT_GOODS_PRICE_median= PreviousApplication['AMT_GOODS_PRICE'].median()
print('mean =',AMT_GOODS_PRICE_mean)
print('median =',AMT_GOODS_PRICE_median)
sns.boxplot(PreviousApplication['AMT_GOODS_PRICE'])
plt.show()


# In[124]:


PreviousApplication['AMT_GOODS_PRICE'].fillna(AMT_GOODS_PRICE_median, inplace=True) ##Replace missing value with median


# In[125]:


list(PreviousApplication.columns[(PreviousApplication.isnull().mean()<=0.40) & (PreviousApplication.isnull().mean()>0)])


# In[126]:


print(PreviousApplication['CNT_PAYMENT'].value_counts())
CNT_PAYMENT_mean= PreviousApplication['CNT_PAYMENT'].mean()
CNT_PAYMENT_median= PreviousApplication['CNT_PAYMENT'].median()
print('mean =',CNT_PAYMENT_mean)
print('median =',CNT_PAYMENT_median)
sns.boxplot(PreviousApplication['CNT_PAYMENT'])
plt.show()


# In[127]:


PreviousApplication['CNT_PAYMENT'].fillna(CNT_PAYMENT_median, inplace=True) ##Replace missing value with median


# In[128]:


list(PreviousApplication.columns[(PreviousApplication.isnull().mean()<=0.40) & (PreviousApplication.isnull().mean()>0)])


# In[129]:


print(PreviousApplication['PRODUCT_COMBINATION'].value_counts())


# In[130]:


PreviousApplication['PRODUCT_COMBINATION'].fillna('Cash', inplace=True) ##Replace missing value with common value


# In[131]:


filter_days = [col for col in PreviousApplication if col.startswith('DAYS')]
print(PreviousApplication[filter_days])


# In[132]:


PreviousApplication[filter_days] = abs(PreviousApplication[filter_days])
print(PreviousApplication[filter_days])


# In[133]:



def uni_pre(var):

    plt.style.use('ggplot')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(20,8))
    
    sns.countplot(x=var, data=PreviousApplication,ax=ax,hue='NAME_CONTRACT_STATUS')
    ax.set_ylabel('Total Counts')
    ax.set_title(f'Distribution of {var}',fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
    
    plt.show()


# In[134]:


# Contract status of previous application
Uni = PreviousApplication["NAME_CONTRACT_STATUS"].value_counts()
pie = pd.DataFrame({'labels': Uni.index,'values': Uni.values})
pie.iplot(kind='pie',labels='labels',values='values', title='Contract status of previous application')


# In[135]:


uni_pre('NAME_CONTRACT_TYPE')


# In[136]:


uni_pre('NAME_CLIENT_TYPE')


# In[137]:


uni_pre('NAME_PAYMENT_TYPE')


# In[138]:


# Was the client old or new client when applying for the previous application
client = PreviousApplication["NAME_CLIENT_TYPE"].value_counts()
df2 = pd.DataFrame({'labels': client.index,'values': client.values})
df2.iplot(kind='pie',labels='labels',values='values', title='Was the client old or new client when applying for the previous application')


# In[139]:


temp = PreviousApplication["NAME_GOODS_CATEGORY"].value_counts()
temp.iplot(kind='bar', xTitle = 'GOODS CATEGORY', yTitle = "Count", title = 'What kind of goods did the client apply for in the previous application', colors=['#75a275'])


# In[140]:


# Replacing XNA with NaN
PreviousApplication = PreviousApplication.replace('XAP',np.NaN)


# In[141]:


# Reasons of previous application rejection
rejection = PreviousApplication["CODE_REJECT_REASON"].value_counts()
rejection.iplot(kind='bar', xTitle = 'Reason', yTitle = "Count", title = 'Reasons of previous application rejection')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # merging application and previous application data 

# In[143]:


## Merging the two files for analysis
Application_merge = pd.merge(Application, PreviousApplication, how='left', on=['SK_ID_CURR'])


# In[144]:


Application_merge.info()


# In[145]:


def plotuni_merge(Varx,Vary):
    
    plt.style.use('ggplot')
    sns.despine
    New_Dat = Application_merge.pivot_table(values='SK_ID_CURR', 
                      index=Varx,
                      columns=Vary,
                      aggfunc='count')
    New_Dat=New_Dat.div(New_Dat.sum(axis=1),axis='rows')*100
    sns.set()
    New_Dat.plot(kind='bar',stacked=True,figsize=(20,8))
    plt.title(f'Effect Of {Varx} on Loan Approval')
    plt.xlabel(f'{Varx}')
    plt.ylabel(f'{Vary}%')
    plt.show()


# In[146]:


plotuni_merge('CODE_GENDER','NAME_CONTRACT_STATUS')


# In[147]:


plotuni_merge('TARGET','NAME_CONTRACT_STATUS')


# In[148]:


plotuni_merge('FLAG_OWN_CAR','NAME_CONTRACT_STATUS')


# # Insights

# # Note

# In[150]:


## Data is highly imbalanced


# # Application Data Insights

# In[149]:


## Adult less likely to default loan & Teenager are more likely to default loan.
## state servant is most likely to may be because state servant job is more stable & commercial associate are non-defaulters are more likely defaulters
## married people are less likely to default loan as compare to other family status applicant.
## Higher education is highly secure were secondary special is risky to give laon
## We can see that high income people are less likely to default loan. May be because there earring is more than loan EMI.


# # Previous Application Data Insights

# In[ ]:


## We can see that maximum consumer get approved  but cash loan face some difficult get approved & refused & cancel also high.
## We can observe from repeater maximum applicant get approved but there are still many refused & cancel applicant. Where New applicant & refreshed approved easily.
## We can observe that maximum client are repeater.
## We can see that maximum loans are for mobile, consumer, computers & audio/video category. We can say majority of loans taken for electronics product.


# In[ ]:


## We see that car ownership doesn't have any effect on application approval or rejection.But we saw earlier that the applicant who has a car has lesser chances of default. The bank can add more weightage to car ownership while approving a loan amount
## We observe that code gender doesn't have any effect on application approval or rejection. But we observe earlier that female have lesser chances of default compared to males.  bank can add more weightage to female while approving a loan amount.
## We observe that the people who were approved for a loan earlier are defaulted less often where as people who were refused a loan earlier have higher chances of defaulting.

