#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
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

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)


#############################################
# Eksik Değer Problemini Çözme
#############################################
"""
Eksik değer problem çözümünde; eğer ağaca dayalı yöntemler kullanılıyorsa bu durumda eksik değerler tıpkı
aykırı değerlerderki gibi gözardı edililebilir durumlardır.
Doğrusal yöntemler, gradient descent gibi yöntemlerde oluduğu gibi değilde daha esnek ve dallara ayırmalı
bir şekilde çalışıyor olduğundan dolayı buradaki aykırılıların ve eksikliklerin etkisi yoka yakındır (ağaca dayalı yöntemlerde)
Tamamen etkisi yok demek doğru olmaz ama yoka yakındır. bir istisnası var eğer ilgilendiğimiz problem regresyon problemi ve 
bağımlı değişkende sayısal bir değişkense bu durumda orada aykırılık olması durumunda
sonuca gitme süresi biraz uzar.
Eksik değer ve aykırı değerlerin nerede ne kadar etkileri var bilmek gerekmektedir.
Doğrusal yöntemler ve gradient descent gibi yöntemlerde bu teknikler çok daha hassas iken 
ağaca dayalı yöntemlerde bunların etkisi daha düşüktür.
"""
missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape  # dropna kullanırken dikkat etmek lazım

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# age i biliyoruz ama elimizde çok fazla sayıda değişlken olursa ne yapacağız
# df.apply(lambda x: x.fillna(x.mean()), axis=0)
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

###################
# Kategorik Değişken Kırılımında Değer Atama
###################


df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# üsttekini daha açık şekilde yapmak isteseydik nasıl yapardık:
df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# üstteki yöntemde bu durumda iki sınıfa sahip olan kategorik değişkenlerin ilk sınfını atacak diğer sınıfını tutacak

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
# bu aşamayla verimizi bir makine öğrenmesi tekiniğini kullanmak için uygun bir hale getiriyoruz

# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) # knn nasıl çalışır? --> bana arkadaşını söyle sana kim olduğunu söyleyeyim der
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# işlem yaptığımız değerleri görmek istedik ama burada problem var:
# 1.aşama: değerleri standartlaştırdık o yüzden geri dönüştürmemiz lazım
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

# yeni değişkenleri eski dataframe bir ata bakalım:
df["age_imputed_knn"] = dff[["Age"]]

# atadıklarını bana bir göster:
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()] # atamadan sonra ne varsa getir bir bakalım


###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma


#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################
# eksiklik bizim için her zaman kötü bir şey midir? aslında birazda bunu inceleyeceğiz
# tüm bunları görselleştirerek bir bakalım:
msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()
# korelasyonunda bir e yakın olması aralarında doğru bir ilişki var demek
# -1 e yakın olması aralarında ters bir ilişki var demek

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)



###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)


