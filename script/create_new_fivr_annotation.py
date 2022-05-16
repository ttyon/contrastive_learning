import glob
import os
import pickle as pk
import numpy as np
import torch
from tqdm import tqdm

dataset_path = glob.glob('/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/fivr/avg_pca1024/' + '*/*.npy')
dataset = []

for d_p in dataset_path:
    dataset.append(os.path.basename(d_p).rsplit('.')[0])
    # dataset.append(os.path.basename(d_p))'

print(len(dataset))

annotation_path = "/workspace/datasets/fivr.pickle"
annotation = pk.load(open(annotation_path, 'rb'))
print(annotation.keys())  # 5k, 200k, annotation

######################################################################## 1
# 5k의 key : queries(50), database(5000)

new_anno_5k = dict()
new_anno_5k_queries = []
new_anno_5k_database = []

# 1. 5k의 query 새로 만들기
for query_5k in tqdm(annotation['5k']['queries']):
    if query_5k in dataset:
        new_anno_5k_queries.append(query_5k)

# 2. 5k의 database 새로 만들기
for database_5k in tqdm(annotation['5k']['database']):
    if database_5k in dataset:
        new_anno_5k_database.append(database_5k)

new_anno_5k['queries'] = new_anno_5k_queries
new_anno_5k['database'] = new_anno_5k_database

######################################################################## 2
# 200k의 key : queries(100), database(225960)

new_anno_200k = dict()
new_anno_200k_queries = []
new_anno_200k_database = []

# 3. 200k의 query 새로 만들기
for query_200k in tqdm(annotation['200k']['queries']):
    if query_200k in dataset:
        new_anno_200k_queries.append(query_200k)

# 4. 200k의 database 새로 만들기
for database_200k in tqdm(annotation['200k']['database']):
    if database_200k in dataset:
        new_anno_200k_database.append(database_200k)

new_anno_200k['queries'] = new_anno_200k_queries
new_anno_200k['database'] = new_anno_200k_database

######################################################################## 3
# keys는 100k의 쿼리 비디오 들 --> 쿼리 비디오 없으면 그거 삭제하면 될 듯 그리고 annotation안에 비디오도 삭제하고..
# 100K 기준 쿼리가 88개만 있다.

new_anno_anno = dict()
for key in tqdm(annotation['annotation'].keys()):
    new_anno_anno_temp = []
    if key in dataset:  # 쿼리 비디오 기준으로 그 비디오 쿼리 키 값이 존재할 때 value에서 없는 비디오 검색
        # print(len(annotation['annotation'][key]))
        new_anno_anno[key] = dict()
        for key_anno in annotation['annotation'][key]:  # key
            # new_anno_anno_temp.append(value)

            new_value = []
            for value in annotation['annotation'][key][key_anno]:
                if value in dataset:
                    new_value.append(value)

            if len(new_value) > 0:
                new_anno_anno[key][key_anno] = new_value
                # print("헿")
    else:  # 쿼리 비디오 기준으로 그 비디오 쿼리 키 값이 존재하지 않을 때 그냥 그 key값을 삭제
        pass

######################################################################## save
new_anno = dict()
new_anno['5k'] = new_anno_5k
new_anno['200k'] = new_anno_200k
new_anno['annotation'] = new_anno_anno

with open('/mldisk/nfs_shared_/js/contrastive_learning/new_fivr.pickle', 'wb') as preset:
    pk.dump(new_anno, preset)

with open('/workspace/datasets/new_fivr.pickle', 'wb') as preset:
    pk.dump(new_anno, preset)
