import glob
import os
import pickle as pk
import numpy as np
import torch
from tqdm import tqdm

dataset_path = glob.glob('/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/vcdb/avg_pca1024/' + '*/*.npy')
dataset = []

for d_p in dataset_path:
    dataset.append(os.path.basename(d_p).rsplit('.')[0])
    # dataset.append(os.path.basename(d_p))

print(len(dataset))

annotation_path = "/workspace/datasets/vcdb.pickle"
annotation = pk.load(open(annotation_path, 'rb'))
print(annotation.keys())  # video_pairs, index, negs

######################################################################## 1
# video_pairs

new_anno_video_pairs = []
for pair in tqdm(annotation['video_pairs']):
    vid1, vid2 = pair['videos'][0], pair['videos'][1]
    if vid1 in dataset and vid2 in dataset:
        new_anno_video_pairs.append(pair)

######################################################################## 2
# index

new_anno_index = []
for idx in tqdm(annotation['index']):
    if idx  in dataset:
        new_anno_index.append(idx)

######################################################################## 3
# negs

new_anno_negs = []
for negs in tqdm(annotation['negs']):
    if negs in dataset:
        new_anno_negs.append(negs)

######################################################################## save
new_anno = dict()
new_anno['video_pairs'] = new_anno_video_pairs
new_anno['index'] = new_anno_index
new_anno['negs'] = new_anno_negs

with open('new_vcdb.pickle', 'wb') as preset:
    pk.dump(new_anno, preset)

