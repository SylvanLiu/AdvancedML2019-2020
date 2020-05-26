__author__ = "SylvanLiu"
__version__ = "1.0.0"

"""AdvML_1.py: a fast regression before the deadline xD."""


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from numpy.linalg import matrix_rank
import math
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from itertools import cycle
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

# Define global variables.
root_path = '/Users/sylvanliu/Desktop/Adv.ML/'
global_year = '2017-01-01'

# Read the data of Human-Freedom-Index from its original csv file.
data_ = pd.read_csv(root_path + 'HFI_Original.csv')

# Generate a rough report of the original data.

# from pandas_profiling import ProfileReport
# ProfileReport(data).to_file(output_file="Report_AvdML.html")

data = pd.DataFrame()
data = data_
data['year'] = pd.to_datetime(data['year'], format='%Y')
data = data.set_index(['year'])
data = data.loc[global_year]
data = data.reset_index()

non_numerical_feature = [1, 2, 3, 4]
# Location of non basic features(the features are calculated by others)
non_basic_feature = [5, 6, 7, 11, 18, 22, 23, 24, 28, 31, 34, 39, 41,
                     45, 46, 54, 58, 60, 61, 62, 68, 70, 81, 87, 90, 93, 98, 99, 103, 110, 117, 118, 119, 120]
# Totally empty columns (are features which are very difficult to acquire)
empty_column = [20, 21, 29, 30, 37, 38, 40, 41, 43, 44]
# Columns need to be ignored.
questionable_feature = [80]
# Columns which are highly correlated to other columns(have correlation coefficient is greater than 0.9)
highly_correlated = [66, 67, 92,
                     35, 39, 49, 56, 57, 11, 9, 10, 8, 17, 22]
# Non basic features but are necessary(theirs basic features are partially or totally missing)
exempted_nbf = [68, 93, 46, 50, 58, 81, 23]
# Concentrate all lists taht hold the elements which are about to be abandoned.

abandon_list = non_basic_feature + empty_column + \
    questionable_feature + highly_correlated + non_numerical_feature

# Remove duplicates.
abandon_list = list(dict.fromkeys(abandon_list))
# Manipulate data into a shape we need.
final_list = []
for location in abandon_list:
    if location not in exempted_nbf:
        final_list.append(data.columns[location-1])
data = data.drop(columns=final_list, axis=1)

# Print the percentage that how many values are missing, then fill them by 0.

# print("{:.8%}".format((data.isnull().sum().sum() /
#                        (data.count(axis=0).sum()*data.count(axis=1).sum()))))


data = data.fillna(0)
# a = np.matrix((data.values), dtype='float')
# print(matrix_rank(a.astype(int)))

""" Export customised data as actual needs. """


def export_customised_csv(data_, abandon_list):
    # Adjustable.
    local_year = '2017-01-01'
    data_c = pd.DataFrame()
    data_c = data_
    data_c['year'] = pd.to_datetime(data_c['year'], format='%Y')
    data_c = data_c.set_index(['year'])
    data_c = data_c.loc[local_year]
    data_c = data_c.reset_index()
    # Adjustable.
    exempted_nbf_ = [68, 93, 46, 50, 58, 81, 23, 5, 6]
    final_list_ = []
    for location in abandon_list:
        if location not in exempted_nbf_:
            final_list_.append(data_c.columns[location-1])
    data_c = data_c.drop(columns=final_list_, axis=1)
    data_c = data_c.set_index(['hf_rank'])
    data_c.sort_values('hf_rank', inplace=True, ascending=True)
    data_c.to_csv(root_path+'HFI_Modified.csv', header=True)


# export_customised_csv(data_, abandon_list)


""" Print the bilingual index after being modified. """


def save_modified_data():
    f = open(root_path + 'AdvML_appendix.txt', 'r')
    lines = f.readlines()
    for column_name in list(data.columns.values):
        for line in lines:
            code_ = line.split(",")[0]
            name_ = line.split(",")[1]
            if column_name == code_:
                print(code_ + ',' + name_)

# save_modified_data()


""" This part visualises the result applying affinity propagation,
use variable methods for decomposition high-dimensional original data. """


def AP_visualisation(axs, labels, cci, data):
    zoom_coefficient = 0.032
    x_mmm = zoom_coefficient * (max(data[:, 0]) - min(data[:, 0]))
    y_mmm = zoom_coefficient * (max(data[:, 1]) - min(data[:, 1]))
    colors = cycle('bgrcmyk')
    for k, col in zip(range(len(cci)), colors):
        member_nums = labels == k
        centre_coor = data[cci[k]]
        # Plot points of members in current cluster.
        axs.plot(data[member_nums, 0],
                 data[member_nums, 1], col + '.', alpha=0.32)
        # Plot point of centre in current cluster.
        axs.plot(centre_coor[0], centre_coor[1], col + 'o', alpha=0.32)
        # Plot line from each member to the centre
        for point in data[member_nums]:
            axs.plot([centre_coor[0], point[0]], [
                     centre_coor[1], point[1]], color=col, alpha=0.32)
        # Plot text next to the centre point.
        axs.text(centre_coor[0]+x_mmm, centre_coor[1]+y_mmm,
                 str(k), fontsize=16, alpha=0.64)
    # Plot text next to each member points.
    i = 0
    for point in data:
        i += 1
        axs.text(point[0]-x_mmm, point[1] - y_mmm,
                 str(i), fontsize=8, alpha=0.64)


# t-SNE has the best average performance, so it will be used for all following operations.
X_TSNE = TSNE(n_components=1, init='random',
              random_state=0).fit_transform(data)


X_PCA = PCA(n_components=1).fit_transform(data)

X_NMF = NMF(n_components=1, init='random', random_state=0).fit_transform(data)


''' fig1 = plt.figure('T-SNE')
ax1 = fig1.gca()

fig2 = plt.figure('PCA')
ax2 = fig2.gca()

fig3 = plt.figure('NMF')
ax3 = fig3.gca()

ax1.hist(X_TSNE, density=True, bins=64, color='blue',alpha = 0.7)
ax2.hist(X_PCA, density=True, bins=64, color='red',alpha = 0.7)
ax3.hist(X_NMF, density=True, bins=64, color='black',alpha = 0.7) '''

MigStock = pd.read_excel(root_path+'UN_MigrantStockTotal.xlsx')
data_temp = pd.DataFrame()
data_temp = data_
data_temp['year'] = pd.to_datetime(data_temp['year'], format='%Y')
data_temp = data_temp.set_index(['year'])
data_temp = data_temp.loc[global_year]
data_temp = data_temp.reset_index()

data_temp['MigStock'] = 0
num_1 = []
a_1 = 0
b_1 = 0
rows_1 = MigStock.iloc[:, 1].values


for i in data_temp['countries']:
    for j in rows_1:
        if i == j:
            data_temp.iloc[a_1, -1] = MigStock.iloc[b_1, 10]
            if MigStock.iloc[b_1, 10] != 0:
                num_1.append(a_1)
            break
        b_1 += 1
    a_1 += 1
    b_1 = 0

# print(data_temp.iloc[num_1]['MigStock'])


POP = pd.read_excel(
    root_path+'WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx')
data_temp['Population'] = 0
num_2 = []
a_2 = 0
b_2 = 0
rows_2 = POP.iloc[:, 2].values

for i in data_temp['countries']:
    for j in rows_2:
        if i == j:
            data_temp.iloc[a_2, -1] = POP.iloc[b_2, 10]
            if POP.iloc[b_2, 10] != 0:
                num_2.append(a_2)
            break
        b_2 += 1
    a_2 += 1
    b_2 = 0

# print(data_temp.iloc[num_2]['Population'])


def temp_1(decom, num_1, num_2, data_temp):
    x_ = []
    y_ = []
    ii = 0
    print(decom)
    for i in num_1:
        # x_.append(data_temp.iloc[i]['MigStock'] /
        #           (data_temp.iloc[i]['Population']*1000))
        x_.append(data_temp.iloc[i]['MigStock'] / 248861296)
        y_.append((decom[:])[ii])
        ii += 1
    return x_, y_

# x_threshold=[0.0, 10],y_threshold=[-10.5, 4] for T-SNE
# x_threshold=[0.0, 0.3],y_threshold=[-20, 21] for PCA
# x_threshold=[0.0, 0.4],y_threshold=[1, 3] doe NMF


def pre_process(x, y, x_threshold=[0.0, 100000], y_threshold=[0, 10]):
    x_new = []
    y_new = []
    i = 0
    for _y in y:
        if _y > y_threshold[0] and _y < y_threshold[1]:
            if x[i] > x_threshold[0] and x[i] < x_threshold[1]:
                y_new.append(_y)
                x_new.append(x[i])
        i += 1
    return (np.asarray(x_new)).reshape(-1, 1), (np.asarray(y_new)).reshape(-1, 1)

# x is the migration proportion, y is the HFI overall score.
x_1, y_1 = temp_1(data_temp['hf_score'], num_1, num_2, data_temp)
x_1, y_1 = pre_process(x_1, y_1)

# X_train, X_test, y_train, y_test = train_test_split(x_1, y_1)

regressor = LinearRegression()
regressor.fit(x_1,y_1)

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
print(regressor.coef_)

fig1 = plt.figure('Mig')
ax1 = fig1.gca()
y_pred = regressor.predict(x_1)
ax1.plot(x_1, y_pred, '-', color='orange', alpha=0.7,linewidth=7)

ax1.scatter(x_1,y_1, color='gray')


GDP = pd.read_csv(
    root_path+'GDP.csv')
data_temp['GDP'] = 0
num_3 = []
a_3 = 0
b_3 = 0
rows_3 = GDP.iloc[:, 1].values

for i in data_temp['ISO_code']:
    for j in rows_3:
        if i == j:
            data_temp.iloc[a_3, -1] = GDP.iloc[b_3, 59]
            if POP.iloc[b_3, 10] != 0:
                num_3.append(a_3)
            break
        b_3 += 1
    a_3 += 1
    b_3 = 0

# print(data_temp.iloc[num_3]['GDP'])

def temp_2(decom, num_3, num_2, data_temp):
    x_ = []
    y_ = []
    ii = 0
    for i in num_3:
        for b in num_2:
            if i == b:
                x_.append(data_temp.iloc[i]['GDP'] /
                        (data_temp.iloc[b]['Population']*1000))
                y_.append((decom[:])[ii])
                break 

        ii += 1
    return x_, y_

# x is the gdp per capita, y is the HFI overall score.
x_2, y_2 = temp_2(data_temp['hf_score'], num_3, num_2, data_temp)
x_2, y_2 = pre_process(x_2, y_2)

regressor = LinearRegression()
regressor.fit(x_2,y_2)

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
print(regressor.coef_)

fig2 = plt.figure('GDP')
ax2 = fig2.gca()
y_pred = regressor.predict(x_2)
ax2.plot(x_2, y_pred, '-', color='skyblue', alpha=0.9,linewidth=7)

ax2.scatter(x_2,y_2, color='crimson')

plt.show()
