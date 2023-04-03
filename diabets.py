import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("diabetes.csv")
df.head()

###### GENEL RESİM ######
print(df.ndim,
      df.shape,
      df.size)
for col in df.columns:
    print(df[col].name,df[col].dtype)



###### DEĞİŞKENLERİ YAKALAMA VE ANALİZ ETME ######    
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
print(cat_cols)
num_cols = [col for col in df.columns if df[col].dtype != "O"]
print(num_cols)
num_but_cat = [col for col in df.columns if (df[col].nunique() < 10) & (df[col].dtype != "O")]
print(num_but_cat)
cat_cols += num_but_cat
print(cat_cols)
num_cols = [col for col in num_cols if col not in num_but_cat]
print(num_cols)

print("###### ÖZET #####")
print(f"Observations: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print(f'num_cols: {len(num_cols)}')
print(f'num_but_cat: {len(num_but_cat)}')



###### HEDEF DEĞİŞKEN ANALİZİ ######
# Hedef değişkenimiz "outcome"
def target_analys (dataframe,target,col):
    if col in cat_cols:
        print(dataframe.groupby(col).agg({target : ["mean"]}))
    elif col in num_cols:
        print(dataframe.groupby(target).agg({col : ["mean"]}))

target_analys(df,"Outcome","Outcome")
target_analys(df,"Outcome","Age")   



###### AYKIRI GÖZLEM ANALİZİ ######     
def outlier (dataframe,col,q1 = 0.25, q3 = 0.75):
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    IQR = q3 - q1
    low = q1 - IQR*1.5
    up = q3 + IQR*1.5
    return low,up

age_outlier = outlier(df,"Age")
print(age_outlier)

def check_out (dataframe,col):
    low,up = outlier(dataframe,col)
    if dataframe[(dataframe[col] > up) | (dataframe[col] < low)].any(axis=None):
        print("True")
        return True
    else:
        print("False")
        return False
    
check_out(df,"Age")

def show_out (dataframe,col):
    low,up = outlier(dataframe,col)
    if dataframe[(dataframe[col] > up) | (dataframe[col] < low)].shape[0] > 10:
         print(dataframe[((dataframe[col] < low) | (dataframe[col] > up))].head()) 
    else:
        print(dataframe[((dataframe[col] < low) | (dataframe[col] > up))])

show_out(df,"Age")



###### EKSİK VERİ ANALİZİ ######
Missing = df.isnull().values.any()

def missing(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    print(na_columns,ratio)

if Missing == True:
    for col in df.columns:
        print(df.isnull().sum())
        missing(df,col)
else:
    print("Don't have any missing values")  



###### KORELASYON ANALİZİ #######

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize' : (12,12)})
sns.heatmap(corr,cmap = "RdBu")
plt.show()

matrix = df.corr().abs() 

upper_triangle = matrix.where(np.triu(np.ones(matrix.shape), k =1 ).astype(np.bool_))

drop_list = [col for col in upper_triangle.columns if any(upper_triangle[col]) > 90 ]
matrix[drop_list]
df.drop(drop_list,axis=1)



###### EKSİK VE AYKIRI DEĞERLER İÇİN GEREKLİ İŞLEMLER ######

def fill_missing (dataframe, col):
    for col in dataframe.columns:
        if dataframe[col].isnull().values.any() == True:
            if col.dtype !="O":
                print(col.fillna(col.mean()))
            elif  col.dtype =="O":  
                print(col.fillna(col.mode()))
            else:
                print(col)
        else:
            ("dont have missing values")       

fill_missing(df,"Age")
fill_missing(df,"BMI")
fill_missing(df,"Outcome")  

print(df.isnull().sum())

df['Insulin'] = df['Insulin'].replace(0, pd.np.nan)
df['Glucose'] = df['Glucose'].replace(0, pd.np.nan)

print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

print(df.isnull().sum())

print(df)

###### ENCODING ######

def rare_analyser(dataframe, target, num_cols):
    for col in num_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe)}),end="\n\n\n")


rare_analyser(df,"Age",num_cols)

def rencoder(dataframe, perc):
    temp_df = dataframe.copy()

    rcolumns = [col for col in temp_df.columns if temp_df[col] .dtype!= "O" and
                    (temp_df[col].value_counts() / len(temp_df) < perc).any(axis=None)]
    
    for variable in rcolumns:
        tmp = temp_df[variable].value_counts() / len(temp_df)
        rlabels = tmp[tmp < perc].index
        temp_df[variable] = np.where(temp_df[variable].isin(rlabels), 'Rare_variable', temp_df[variable])

    return temp_df

new_df = rencoder(df, 0.05)
rare_analyser(new_df,"Age",num_cols)
 
for col in df.columns:
    print(df[col].name,df[col].dtype)



###### STANDARTLAŞTIRMA ######

print(df.head())

ss = StandardScaler()

for col in df.columns:
    df["NEW_" + col] = ss.fit_transform(df[[col]])

print(df.head())


###### MODEL ######

y = df["Pregnancies"]
X = df.drop("Pregnancies", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
