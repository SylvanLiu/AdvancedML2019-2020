__author__ = "SylvanLiu"
__version__ = "1.0.1"

"""AdvML_1.py: Generate fundamental visualisation report on raw-data-level as well as rough correlation analysis on the feature-level;;
Pre-process the original data (Reform the index and remove all useless columns);
Roughly applying different decomposition and k-value-not-required clustering methods to the processed data,
subsequently demonstrate the performance of those methods by visualisation.
The relation between k-value and the performance of k-means has also been shown."""

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

# Location of non basic features(the features are calculated by others)
non_basic_feature = [1, 2, 3, 4, 5, 6, 7, 11, 18, 22, 23, 24, 28, 31, 34, 39, 41,
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
    questionable_feature + highly_correlated
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




# sns.scatterplot(X_TSNE[:,0], X_TSNE[:,1])
# plt.show()

# X_PCA = PCA(n_components=2).fit_transform(data)

# sns.scatterplot(X_PCA[:,0], X_PCA[:,1])
# plt.show()

# X_NMF = NMF(n_components=2, init='random', random_state=0).fit_transform(data)

# sns.scatterplot(X_NMF[:,0], X_NMF[:,1])
# plt.show()

# clustering_DBSCAN = DBSCAN().fit(X_NMF)
# L_DBSCAN = clustering_DBSCAN.labels_
# clustering_OPTICS = OPTICS(min_samples=2).fit(X_NMF)
# L_OPTICS = clustering_OPTICS.labels_
# clustering_AC = AgglomerativeClustering().fit(X_NMF)
# L_AC = clustering_AC.labels_

# clustering_AP_TSNE = AffinityPropagation().fit(X_TSNE)
# L_AP_TSNE = clustering_AP_TSNE.labels_
# CCI_AP_TSNE = clustering_AP_TSNE.cluster_centers_indices_
# clustering_AP_PCA = AffinityPropagation().fit(X_PCA)
# L_AP_PCA = clustering_AP_PCA.labels_
# CCI_AP_PCA = clustering_AP_PCA.cluster_centers_indices_
# clustering_AP_NMF = AffinityPropagation().fit(X_NMF)
# L_AP_NMF = clustering_AP_NMF.labels_
# CCI_AP_NMF = clustering_AP_NMF.cluster_centers_indices_

# Draw the graph while function above uses the data from different decomposition methods,
# namely T-SNE, PCA, and NMF, hence we can have a glence at which one is the best.

# fig_1, axs_1 = plt.subplots(1, 1, constrained_layout=True)
# plot_kwds = {'color': 'k', 'alpha': 0.32, 's': 64, 'linewidths': 0}
# axs_1[0].scatter(X_TSNE[:, 0], X_TSNE[:, 1], **plot_kwds)
# # axs_1[1].scatter(X_PCA[:, 0], X_PCA[:, 1], **plot_kwds)
# # axs_1[2].scatter(X_NMF[:, 0], X_NMF[:, 1], **plot_kwds)
# AP_visualisation(axs_1[0], L_AP_TSNE, CCI_AP_TSNE, X_TSNE)
# fig_1.axes[0].get_xaxis().set_visible(False)
# fig_1.axes[0].get_yaxis().set_visible(False)
# axs_1[0].set_title('TSNE')
# AP_visualisation(axs_1[1], L_AP_PCA, CCI_AP_PCA, X_PCA)
# fig_1.axes[1].get_xaxis().set_visible(False)
# fig_1.axes[1].get_yaxis().set_visible(False)
# axs_1[1].set_title('PCA')
# AP_visualisation(axs_1[2], L_AP_NMF, CCI_AP_NMF, X_NMF)
# fig_1.axes[2].get_xaxis().set_visible(False)
# fig_1.axes[2].get_yaxis().set_visible(False)
# axs_1[2].set_title('NMF')

while 1:
    # t-SNE has the best average performance, so it will be used for all following operations.
    X_TSNE = TSNE(n_components=2, init='random',
                random_state=0).fit_transform(data)


    """ This part export the original data(no columns are removed) as csv file,
    which has been ranked ascendingly according to the clustering result above."""


    def export_data_with_labels(labels_):
        data_['year'] = pd.to_datetime(data_['year'], format='%Y')
        data_ = data_.set_index(['year'])
        data_ = data_.loc[global_year]
        data_ = data_.reset_index()
        data_['labels'] = labels_
        data_ = data_.set_index(['labels'])
        data_.sort_values('labels', inplace=True, ascending=True)
        data_.to_csv(root_path+'HFI_Clustered.csv', header=True)

    # export_data_with_labels(L_AP_TSNE)


    """ This part draws the line chart while k represents the x-axis, and indices 
    'averge distance of each points to their cluster centres' as well as 'silhouette score'
    represent the y-axis respectively, which can demonstrate the noteworthy k value """


    performance_x = []
    performance_ss = []
    performance_ad = []
    # Evaluate the performance of k-means while iteratively changing the number of cluster centres,


    def evaluate_kmeas(X_):
        for n_clusters in range(2, 16+1):
            clustering_KMEANS = KMeans(n_clusters=n_clusters)
            L_KMEANS = clustering_KMEANS.fit_predict(X_)
            ss = silhouette_score(X_, L_KMEANS)
            averge_distance = math.sqrt(
                clustering_KMEANS.inertia_/X_.shape[0])
            performance_x.append(n_clusters)
            performance_ss.append(ss)
            performance_ad.append(averge_distance)


    evaluate_kmeas(X_TSNE)
    # Align both lines from their start point to their end point.
    amplify_coefficient = (max(performance_ad)-min(performance_ad)) / \
        (max(performance_ss)-min(performance_ss))
    translation_coefficient = performance_ss[0] * \
        amplify_coefficient - performance_ad[0]
    ii = 0
    for ss in performance_ss:
        performance_ss[ii] = ss*amplify_coefficient - translation_coefficient
        ii += 1

    # Draw line chart of k-averge_distance.
    fig_2, axs_2 = plt.subplots(1, 1, constrained_layout=True)
    performance_ad = pd.DataFrame(
        {'k': performance_x, 'averge_distance': performance_ad})
    axs_2.plot('k', 'averge_distance', data=performance_ad,
            color='skyblue', label='averge_distance/k',linewidth=4)
    # Draw line chart of k-silhouette_score.
    performance_ss = pd.DataFrame(
        {'k': performance_x, 'silhouette_score': performance_ss})
    axs_2.plot('k', 'silhouette_score', data=performance_ss,
            color='orangered', label='silhouette_score/k',linewidth=4)
    plt.legend(fontsize='x-large')
    plt.grid()
    plt.show()
