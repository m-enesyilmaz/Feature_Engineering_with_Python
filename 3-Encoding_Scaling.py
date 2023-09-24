#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None) #bütün sütunları göster
pd.set_option('display.max_rows', None)  #bütün satırları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # virgülden sonra 3 basamak göster
pd.set_option('display.width', 500) # genişlik 500 olsun

pd.concat()

#############################################
# Label Encoding & Binary Encoding
#############################################

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

# ordinal bir değişken -> sıralı bir değişkendir yani sınıflar arası fark var (büyük olan değişken, küçük olan değişken)
# eğer değişken sıralı değilse label encoding yanlış olabilir
df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

# iki sınıflı kategorik değişkenleri bulmak istersek:
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

# iki sınıflı kategorik değişkenleri bulmak istersek:
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

#2.yol
#binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

df[binary_cols].head()


for col in binary_cols:
    label_encoder(df, col)
# df[binary_cols].head()
# burada da görüldüğü gibi ikili olanları yakalamış ama na olanları unique değer gibi görüp doldurmuş
# o yüzden 2 de gösteriyor

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())  # burada da görüldüğü gibi na değerini eşsiz (unique) değer olarak görmüş

#############################################
# One-Hot Encoding
#############################################

df = load()
df.head()
df["Embarked"].value_counts()
# dummie (kukla) değişken tuzağı drop_first = True yaparak kurtulabilriz ama değişkenler
# birbirinden etkileniyorsa sonuçları bozabilir (yüksek korelasyona sebep olabilir yanlış anlamladırılabilir)
# bu yüzden ilk sınf drop edilir değişkenlerin birbiri üzerinden oluşturulmasının önüne geçilmeye çalışılır
pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


one_hot_encoder(df, ohe_cols).head()

df.head()

#############################################
# Rare Encoding
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# 1. Sınıf Frekansı
# 2. Sınıf Oranı
# 3. Sınıfların target açısından değerlendirilmesi

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()


#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

# gradient descent gibi yöntemlerde model süresini kısaltmak için kullanılır (daha kısa sürede minimum noktaya ulaşır)
# uzaklık temelli (knn vb gibi) algoritmalarda ya da benzerlik benzemezlik gibi yöntemlerde ölçeklerin
# birbirinden farklı olması durumu benzerlik benzemezlik hesaplarında yanlılığa sebep olmaktadır
# istinasna var mı (scaling gerektirmeyen) var --> ağaca dayalı yöntemlerin bir çoğu eksik değerden aykırı değerden
# etkilenmez, standartlaştırmalardan etkilenmez

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

#2. yol
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
scaler = StandardScaler().fit(df[["Fare"]])
df["Fare"] = scaler.transform(df[["Fare"]])
df["Fare"].describe().T


###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################
# RobustScaler: z = (x - median) / IQR

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in age_cols:
    num_summary(df, col, plot=True)

# bilginin dağılımı aynı duruyor biz sadece bilginin ifade ediliş şeklini değiştiriyoruz
# scaling ile bilgimizin ifade ediliş biçimini algotritmalarda kullanıma uygun hale getrimiş oluyoruz aslında

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)
# qcut metodu bir değişenin değerleini küçükten büyüğe sıralar ve çeyrek değerler göre istenilen parçaya böler


