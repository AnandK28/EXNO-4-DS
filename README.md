![image](https://github.com/user-attachments/assets/9295037d-9373-40e2-b0ec-90796757d229)# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
BMI.csv
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/b10c5ddb-5a2e-4960-acd3-c0f3a1e42800)

```
df.dropna()
```
![image](https://github.com/user-attachments/assets/cef48892-354f-4cd5-a535-94f2a97cd1be)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/627d5f1e-651d-4048-91fb-d05c1b4c36e5)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/8698d552-df16-499e-90d3-5728e0c8338a)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/7f5531a0-d6de-4fb5-b361-c461fd900924)

```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![image](https://github.com/user-attachments/assets/5bfad717-1a4e-4313-afff-df59b32e544c)

```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/a02b8993-ca26-403f-bd64-52a0ccd5c76c)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/aa865cea-9b40-45b6-a1c1-7e3aa68bded2)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/b6845de5-0e66-446f-969d-e4f475981e11)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/27ba7380-d6cd-4975-864d-0b200ec63ec2)

```
chip2,p, _, _=chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chip2}")
print(f"P-value: {p}")
```
![image](https://github.com/user-attachments/assets/7a3c47e7-e4d2-43c1-9e52-a6dc01d80279)

```
import pandas as pd 
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif 

data = { 
'Feature1': [1, 2, 3, 4, 5], 
'Feature2': ['A', 'B', 'C', 'A', 'B'], 
'Feature3': [0, 1, 1, 0, 1], 
'Target': [0, 1, 1, 0, 1] 
} 
df = pd.DataFrame(data) 

x= df[['Feature1', 'Feature3']] 
y = df['Target'] 
 
selector = SelectKBest(score_func=mutual_info_classif, k=1) 
X_new = selector.fit_transform(x, y)

selected_feature_indices = selector.get_support(indices=True) 


selected_features = X.columns[selected_feature_indices] 
print("Selected Features:") 
print(selected_features) 
```
![image](https://github.com/user-attachments/assets/b28e63c4-d423-4172-b8d9-c797576f2332)


Income.csv
```python
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/054a02f2-86ce-4822-9fb4-8cf1061f2217)


```python

data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/8e1aed87-efce-42e7-abfa-55106e3f0e1a)

```python

missing=data[data.isnull().any(axis=1)]
missing
```![image](https://github.com/user-attachments/assets/b65ce1db-43c8-467a-addf-a4843cfa12f3)

```python

data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/f8d59d7b-5c5c-44cb-ac80-1c23d5303fc4)

```python
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/3976720d-29b9-45ef-997f-c3baaa4a14a6)

```python
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/856c908a-037a-4105-b4b3-600b53dc8f27)

```python


data2
```
![image](https://github.com/user-attachments/assets/5d97f411-6a3e-4596-b64d-7b590b313dca)

```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/ce457d71-2522-4da4-8d95-4888e92bb20b)

```python

columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/9b6cd790-a67e-4b2f-9098-7e44a8826154)

```python


features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/51302c5e-2720-4a26-afe3-8b4700dcdd10)

```python
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/8be31873-5aea-4d8c-b1cb-3ca07a6ba7bf)

```python

x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/989742ef-ef02-4d6e-a4d5-a854f4014f15)

```python

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/3d80de2e-8fbe-41b2-a3b0-534e144cb5ac)

```python

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/da25db88-744b-4cc5-95c5-045c76bfe513)

```python

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/21d7b0b1-329b-414c-af22-291d64bbf4ee)

```python

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/c3dd3f25-f517-4046-8458-5aeff13e5f7e)

```python

data.shape
```
![image](https://github.com/user-attachments/assets/a6edbcd1-0de5-47c6-8199-19ec203529b8)

```python

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/5f0d27e6-1b29-4c9e-b95d-4a01f93b0eb3)

```python

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/119538b7-d862-4f34-b5e7-0d3e45ea36a0)

```python

tips.time.unique()
```
![image](https://github.com/user-attachments/assets/7cd8552b-a71a-4220-9278-ae900fbee920)

```python

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/dc81baf6-70ee-4774-85b5-ff20fd1c85e9)

```python

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/1806caf0-4591-4d2e-a010-9efe26057b04)


# RESULT:
Thus,The given data is read and performed Feature Scaling and Feature Selection process and saved the
data to a file.
