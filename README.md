# EXNO:4-DS
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

# RESULT:
Thus,The given data is read and performed Feature Scaling and Feature Selection process and saved the
data to a file.
