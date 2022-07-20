#!/usr/bin/env python3
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30,
             ha="right", rotation_mode="anchor")
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


layers_temp = []
neurons_temp = []
means_temp = []
standard_temp = []

with open('AppError_data_2(1layer).txt', newline = '') as data:                                                                                          
		data_array = csv.reader(data, delimiter=',')
		for data in data_array:
			layers_temp.append(data[0])
			neurons_temp.append(data[1])
			means_temp.append(data[2])
			standard_temp.append(data[3])

layers = (np.arange(int(layers_temp[0]), int(layers_temp[-1])+1)).tolist()
neurons = (np.arange(int(neurons_temp[0]), int(neurons_temp[-1])+1, 10)).tolist()

means = np.zeros((len(layers), len(neurons)))
standard_var = np.zeros((len(layers),len(neurons)))

k = 0
for i in range(len(layers)):
	for j in range(len(neurons)):
		means[i, j] = means_temp[k]
		standard_var[i, j] = standard_temp[k]
		k = k+1

savePath_Plot_mean = "AppError_Mean_"+str(layers[-1])+'x'+str(neurons[-1])+'.png'
savePath_Plot_stdvar = "AppError_StdVar_"+str(layers[-1])+'x'+str(neurons[-1])+'.png'

fig, ax = plt.subplots(figsize=(8, 8))
im, cbar = heatmap(means, layers, neurons,
                   ax=ax, cmap="BuPu", cbarlabel="mean")
texts = annotate_heatmap(im, valfmt="{x:.4f}")
fig.tight_layout()
#plt.show()
plt.savefig(savePath_Plot_mean)


fig, ax = plt.subplots(figsize=(8, 8))
im, cbar = heatmap(standard_var, layers, neurons,
                   ax=ax, cmap="BuPu", cbarlabel="standard var")
texts = annotate_heatmap(im, valfmt="{x:.4f}")
fig.tight_layout()
#plt.show()
plt.savefig(savePath_Plot_stdvar)

"""
fig, ax = plt.subplots(1, 2)
im, cbar = heatmap(means, layers, neurons,
                   ax=ax[0], cmap="viridis", cbarlabel="mean")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
im2, cbar2 = heatmap(standard_var, layers, neurons,
                     ax=ax[1], cmap="viridis", cbarlabel="standard_variation")
texts2 = annotate_heatmap(im2, valfmt="{x:.2f}")
fig.tight_layout()
#plt.show()
plt.savefig(savePath_Plot)
"""
