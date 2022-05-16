import glob
import os
import pickle as pk
import numpy as np
import torch
from tqdm import tqdm

# annotation = pk.load(open('/mldisk/nfs_shared_/js/contrastive_learning/new_vcdb.pickle', 'rb'))
#
# print(len(annotation['video_pairs']))
#
# a = np.load("/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/vcdb/rmac/core/00274a923e13506819bd273c694d10cfa07ce1ec.npy")
# print("a.shape :", a.shape)
#
# b = np.load("/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/vcdb/rmac_pca1024/core/00274a923e13506819bd273c694d10cfa07ce1ec.npy")
# print("b.shape :", b.shape)


a = np.load("/workspace/pre_processing/pca_params_fivr_mobilenet_avg_1280.npz")

print(a.keys)