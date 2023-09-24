#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
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

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
df.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()


#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["Age"])
plt.show()
# elimizde bir sayısal değişken varsa kutu grafikten sonra
# yaygın kullanılan grafik türü histogram'dır.

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)]  #bunlar aykırı değerlermiş

df[(df["Age"] < low) | (df["Age"] > up)].index #aykırı değerlerin index'lerini getiriyoruz

###################
# Aykırı Değer Var mı Yok mu?
###################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)
df[(df["Age"] < low)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()

df[(df["Fare"] < low) | (df["Fare"] > up)].index


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")

###################
# grab_col_names
###################

dff = load_application_train()
dff.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # üst satırda yaptığımız kategorik değişkenleri buluyoruz.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    # üst satırda yaptığımız değişken nümerik (sayı) görünümlü ama aslında kategorik olanları bul
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    # üst satırda yaptığımız kategorik değişken ama bizim için ölçülebilirliği yok kardinal değişken demiş oluyoruz
    # benim gözlem sayım kadar senin unique sınıfın varsa buradan ne bilgisi çıkacak ki yorumunda bulunuyoruz
    cat_cols = cat_cols + num_but_cat
    # üst satıda ise kategorik olanları yeniden oluşturuyoruz nümerik görünümlü olan kategorikleri de ekliyoruz
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # üst satırda da kategorik görünümlü ama kardinallerde olmayanları seç diyoruz

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # üst satırda da object olmayanları yani int float olanları getir diyoruz
    num_cols = [col for col in num_cols if col not in num_but_cat]
    # numerik görünümlü ama kategorik olanlar vardı onları da çıkar diyoruz
    # ÖNEMLİ: ölçüm niteliği taşıyorsa kategorik bir değişken --> not: verilerin türlerine dikkatle bakmak lazım

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))
    # aykırı değer var mı yok mu diye sordu


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))
    # aykırı değer var mı yok mu diye sordu

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")

grab_outliers(df, "Age", True)

age_index = grab_outliers(df, "Age", True)


outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape  # burada tilda ~ ile outlier olmayanları getir dedik

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[(df["Fare"] < low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


###################
# Recap
###################
df = load()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")


#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

# 17, 3 --> örneğin 17 yaşında olup 3 kere evlenmiş olmak aykırı değerdir
# |--> burada ayrı ayrı her biri aykırı olmaya bilir ama ikisi birlikte aykırı değerdir

# LOF (local outlier factor) gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak
# buna göre aykırı değer tanımı yapabilmemizi sağlar
#  lof sonucu 1 e yakın değerler inlier'dır 1 den uzaklaştıkça outlier olma ihtimali artar diyebiliriz
"""
Örneğin elimizde 100 tane değişken var biz bunun 2 boyutta görselleştirmek istiyoruz. Nasıl yapabiliriz?

Elimizdeki 100 değişkeni o 100 değişkenin taşıdığı bilginin büyük bir kısmını 2 boyuta indirebilirsek bu durumda yapabiliriz.
Bunu da temel bileşen analizi (PCA) yöntemi ile yapabiliriz. 

Bu bir mülakat sorusu olabilir.
"""

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape
for col in df.columns:
    print(col, check_outlier(df, col))


low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape
# tek başına baktığımızda çok fazla aykırılık geldi bir de buna çok değişkenle bakalım

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores   --> isteyenler için sonuçları pozitif te inceleyebiliriz
np.sort(df_scores)[0:5]
# buraya baktığımızda threshold değeri nasıl belirleyeceğiz peki?
# bunu da görselleştirip dirsek yöntemi denilen yönteme göre bakabilir ve ayarlayabiliriz:
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
# buralar yorum tabi ki
th = np.sort(df_scores)[3]

df[df_scores < th]

df[df_scores < th].shape

# bu değerler neden aykırılığa takıldı gözlemleyelim:
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# gözlem sayısı çok olduğunda baskılama yöntemi bizi hatalı sonuçlara götürebilir
# gözlem sayısı az olduğunda ise daha işe yarar snuçlar elde edebiliriz

# ağaç yönteleri kullanıyorsak aykırı değerlere dokunmamyı tercih ediyoruz
# illaki dokunmak istersek de çok ucundan traşlayabiliriz. (iqr değerlerini 99 a 1 yada 95 e 5 lik olarak gibi)
# doğrusal yöntemler kullanıyorsak aykırı değer problemi bizim için tüm ciddiyetiyle devam ediyor olacak
# bu aykırı değerleri ise az sayıdaysa doldurmaktan ziyade silmek tercih edilebilir

