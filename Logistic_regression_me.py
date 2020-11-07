#!/usr/bin/env python
# coding: utf-8

# In[173]:


import pandas as pd
import numpy as np


# In[174]:


churn_data = pd.read_csv(r"C:\Users\Administrator\Desktop\ML\UPGRAD\Logistic Regression\churn_data.csv")


# In[175]:


customer_data = pd.read_csv(r"C:\Users\Administrator\Desktop\ML\UPGRAD\Logistic Regression\customer_data.csv")


# In[176]:


internet_data = pd.read_csv(r"C:\Users\Administrator\Desktop\ML\UPGRAD\Logistic Regression\internet_data.csv")


# In[177]:


df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')


# In[178]:


telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')


# In[179]:


varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
telecom[varlist] = telecom[varlist].apply(binary_map)


# In[180]:


telecom.head()


# In[181]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(telecom[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)

# Adding the results to the master dataframe
telecom = pd.concat([telecom, dummy1], axis=1)


# In[182]:


telecom.head()


# In[183]:


ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')
# Dropping MultipleLines_No phone service column
ml1 = ml.drop(['MultipleLines_No phone service'], 1)
#Adding the results to the master dataframe|
telecom = pd.concat([telecom,ml1], axis=1)

# Creating dummy variables for the variable 'OnlineSecurity'.
os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,os1], axis=1)

# Creating dummy variables for the variable 'OnlineBackup'.
ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ob1], axis=1)

# Creating dummy variables for the variable 'DeviceProtection'. 
dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,dp1], axis=1)

# Creating dummy variables for the variable 'TechSupport'. 
ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ts1], axis=1)

# Creating dummy variables for the variable 'StreamingTV'.
st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,st1], axis=1)

# Creating dummy variables for the variable 'StreamingMovies'. 
sm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,sm1], axis=1)


# In[184]:


telecom.head()


# In[185]:


telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)


# In[186]:


telecom.head()


# In[187]:


telecom['TotalCharges'] = pd.to_numeric(telecom['TotalCharges'],errors='coerce')


# In[188]:


telecom.info()


# In[189]:


num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]


# In[190]:


num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[191]:


telecom.isnull().sum()


# In[192]:


round(100*telecom.isnull().sum()/len(telecom.index),2)


# In[193]:


telecom=telecom[~np.isnan(telecom['TotalCharges'])]


# In[194]:


round(100*telecom.isnull().sum()/len(telecom.index),2)


# ##Spliting the data into training and testing data
# 

# # model building

# In[195]:


from sklearn.model_selection import train_test_split


# In[196]:


x=telecom.drop(['Churn','customerID'],axis=1)
y=telecom['Churn']


# In[197]:


X_train,X_test,y_train,y_test=train_test_split(x,y, train_size=0.7, test_size=0.3, random_state=100)


# ## scaling the model using standard scaler

# In[198]:


from sklearn.preprocessing import StandardScaler


# In[199]:


scaler = StandardScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.describe()


# In[200]:


X_train.describe()


# #### Checking the churn rate is to find out the churn rate is balanced or imbalanced
# 

# In[201]:


churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100
churn


# #### Ploting the correlations between the features

# In[202]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[203]:


plt.figure(figsize=(20,10))
sns.heatmap(telecom.corr(),annot=True)
plt.show()


# ### Droping the features that are having high correalation

# In[204]:


X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                       'StreamingTV_No','StreamingMovies_No'], 1)
X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                         'StreamingTV_No','StreamingMovies_No'], 1)


# In[205]:


X_test.head()


# In[206]:


import statsmodels.api as sm


# In[207]:


lgr=sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial()).fit()
lgr.summary()


# ### Feature Elimination using RFE

# In[208]:


from sklearn.linear_model import LogisticRegression
lgs=LogisticRegression()


# In[209]:


from sklearn.feature_selection import RFE
rfe=RFE(lgs,15)
rfe.fit(X_train,y_train)


# In[210]:


rfe.support_


# In[211]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[212]:


col=X_train.columns[rfe.support_]
col


# In[213]:


X_train.columns[~rfe.support_]


# In[214]:


X_train_sm=sm.add_constant(X_train[col])
X_train_sm.head()


# In[215]:


lgs2=sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
res=lgs2.fit()
res.summary()


# In[216]:


y_train_pred=res.predict(X_train_sm)
y_train_pred


# In[217]:


y_train_pred_final=pd.DataFrame({'Churn':y_train.values,'Churn_prob':y_train_pred})
y_train_pred_final['Cust_id']=y_train.index


# In[ ]:





# In[218]:


y_train_pred_final['Pred_churn']=y_train_pred_final.Churn_prob.map(lambda x:1 if x >0.5 else 0)


# In[219]:


y_train_pred_final.head()


# ### Checking the accuracy of the model using the confusion matrix

# In[220]:


from sklearn import metrics


# In[221]:


confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.Pred_churn )
print(confusion)


# In[222]:


accuracy=metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.Pred_churn)
print(accuracy)


# ### Manual feature elimination using VIF and P values

# In[223]:


plt.figure(figsize=(20,10))
sns.heatmap(X_train_sm.corr(),annot=True)
plt.show()


# #### The Heat map shows that still there are correlations between the features and we need to remove it 

# In[224]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[225]:


vif=pd.DataFrame()
vif["columns"]=X_train[col].columns
vif['VIF']=[variance_inflation_factor(X_train[col].values,i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
vif


# In[226]:


col1=col.drop('MonthlyCharges',1)


# In[227]:


col1


# In[228]:


X_train_sm=sm.add_constant(X_train[col1])
X_train_sm.head()
lgs2=sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
res=lgs2.fit()
res.summary()


# In[229]:


y_train_pred=res.predict(X_train_sm)
y_train_pred

y_train_pred_final=pd.DataFrame({'Churn':y_train.values,'Churn_prob':y_train_pred})
y_train_pred_final['Cust_id']=y_train.index

y_train_pred_final['Pred_churn']=y_train_pred_final.Churn_prob.map(lambda x:1 if x >0.5 else 0)
print(y_train_pred_final)
accuracy=metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.Pred_churn)
print(accuracy)


# In[230]:


vif=pd.DataFrame()
vif["columns"]=X_train[col1].columns
vif['VIF']=[variance_inflation_factor(X_train[col1].values,i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
vif


# In[231]:


col1=col1.drop('TotalCharges',1)
col1


# In[232]:


X_train_sm=sm.add_constant(X_train[col1])
X_train_sm.head()
lgs2=sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
res=lgs2.fit()
res.summary()


# In[233]:


y_train_pred=res.predict(X_train_sm)
y_train_pred

y_train_pred_final=pd.DataFrame({'Churn':y_train.values,'Churn_prob':y_train_pred})
y_train_pred_final['Cust_id']=y_train.index

y_train_pred_final['Pred_churn']=y_train_pred_final.Churn_prob.map(lambda x:1 if x >0.5 else 0)
y_train_pred_final.head()
accuracy=metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.Pred_churn)
print(accuracy)


# In[234]:


vif=pd.DataFrame()
vif["columns"]=X_train[col1].columns
vif['VIF']=[variance_inflation_factor(X_train[col1].values,i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[235]:


TN=confusion[0,0]
TP=confusion[1,1]
FN=confusion[1,0]
FP=confusion[0,1]


# In[236]:


Sensitivity=TP/float(TP+FN)
print(Sensitivity)


# In[237]:


Specificity=TN/float(TN+FP)
print(Specificity)


# In[238]:


# positive predictive value /Precision
print (TP / float(TP+FP))


# In[239]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### Ploting the ROC curve 

# In[240]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

# Calling the function
draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_prob)


# ### Finding the optimal Threshold

# In[241]:


numbers=[float(x)/10 for x in range(10)]
for i in numbers:
     y_train_pred_final[i]=y_train_pred_final.Churn_prob.map(lambda x:1 if x>i else 0)
y_train_pred_final.head()        


# In[242]:


cutoff=pd.DataFrame(columns=['Prob','Accuracy','Sensitivity','Specificity'])
for i in numbers:
    cm1=metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final[i])
    total=sum(sum(cm1))
    Acc=(cm1[0,0]+cm1[1,1])/total
    Speci=cm1[0,0]/(cm1[0,0]+cm1[0,1])
    Sensi=cm1[1,1]/(cm1[1,0]+cm1[1,1])
    
    cutoff.loc[i]=[i,Acc,Sensi,Speci]
print(cutoff)   


# In[243]:


cutoff.plot.line(x='Prob', y=['Accuracy','Sensitivity','Specificity'])
plt.show()


# ### From the plot we can derive that 0.3 is the optimum threshold

# In[244]:


y_train_pred_final['Pred_churn']=y_train_pred_final.Churn_prob.map(lambda x:1 if x >0.3 else 0)
y_train_pred_final.head()


# In[245]:


metric1=metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.Pred_churn)
metric1


# In[246]:


recall=metrics.recall_score(y_train_pred_final.Churn,y_train_pred_final.Pred_churn)
precision=metrics.precision_score(y_train_pred_final.Churn,y_train_pred_final.Pred_churn)
print("Recall =",recall)
print("Precision =",precision)


# In[247]:


p, r, thresholds=metrics.precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### From the Precision_Recall_Curve we can conclude that the optimum threshold is around 0.42
# #### It is our choice according to the business needs which one choose as a threshold [Sensitivity,Specificity] or [Precision. Recall]

# In[ ]:





# ## Making Predictions on using Test Data
# 

# In[248]:


X_test.head()


# In[ ]:





# In[249]:


X_test.head()


# In[254]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])
X_test.describe()


# ###  Noted some ambiguity while scaling the test data.
# ####  Ambiguity not solved [NOTFITTED ERROR]

# In[258]:


X_test=X_test[col1]


# In[260]:


X_test.describe()


# In[273]:


X_test_sm=sm.add_constant(X_test)


# In[274]:


y_test_pred=res.predict(X_test_sm)


# In[288]:


Check=pd.DataFrame()
Check['churn']=y_test
Check['cus_ID']=y_test.index
Check['Predict_prob']=y_test_pred
Check['pred_churn']=Check.Predict_prob.map(lambda x:1 if x>0.3 else 0)
Check.head()


# In[290]:


final_confusion=metrics.confusion_matrix(Check.churn,Check.pred_churn)
final_confusion


# In[292]:


Accuracy=metrics.accuracy_score(Check.churn,Check.pred_churn)
Accuracy


# In[293]:


TN=final_confusion[0,0]
TP=final_confusion[1,1]
FP=final_confusion[0,1]
FN=final_confusion[1,0]


# In[294]:


Specificity=TN/float(TN+FP)
Specificity


# In[295]:


Sensitivity=TP/float(TP+FN)
Sensitivity


# In[296]:


FPR=FP/float(TN+FP)
FPR


# In[298]:


Precision=metrics.precision_score(Check.churn,Check.pred_churn)
Precision


# In[ ]:




