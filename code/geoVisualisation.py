import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gp
import pandas as pd

root_path = '/Users/sylvanliu/Desktop/Adv.ML/'
fig, ax1 = plt.subplots(1, 1)
# Initialise world.
world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))
# Draw background.
world.boundary.plot(ax=ax1, color='#000000')
world.plot(ax=ax1, color='#FFFFFF')

clustered_data = pd.read_csv(root_path+'HFI_Clustered.csv')

colors = ['#F7BE16', '#F78259', '#EB4559', '#844685',
          '#42E6A4', '#216353', '#10375C', '#E7D39F']
i = 0
# 8 is the amount of clusters.
for i, color in zip(range(8), colors):
    indices = clustered_data[clustered_data['labels']
                             == i]['ISO_code'].values.tolist()
    indices_ = iter(indices)
    for index in indices_:
        # There are some very tiny countries don't exist in the plotting library, it will return warning.
        world[world['iso_a3'] == index].plot(ax=ax1, color=color)

handles = [mpatches.Patch(color='#F7BE16', label='Cluster 1'), mpatches.Patch(color='#F78259', label='Cluster 2'), mpatches.Patch(color='#EB4559', label='Cluster 3'), mpatches.Patch(color='#844685', label='Cluster 4'), mpatches.Patch(
    color='#42E6A4', label='Cluster 5'), mpatches.Patch(color='#216353', label='Cluster 6'), mpatches.Patch(color='#10375C', label='Cluster 7'), mpatches.Patch(color='#E7D39F', label='Cluster 8')]
plt.legend(handles=handles)
ax1.set_axis_off()
plt.show()
