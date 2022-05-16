import glob
import os
import pickle as pk

import h5py
import numpy as np
from tqdm import tqdm


def get_feature_list(root='~/datasets/', dataset='vcdb', feat='imac'):
    if dataset == 'vcdb':
        dataset = 'vcdb'
    elif dataset == 'ccweb':
        dataset = 'CC_WEB_VIDEO'
    elif dataset == 'fivr':
        dataset = 'fivr'
    if dataset == 'vcdb':
        return sorted(glob.glob(f'/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/vcdb/{feat}/core/*.npy')) + sorted(glob.glob(f'/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/vcdb/{feat}/distraction/*.npy'))

    print(f'/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/{dataset}/{feat}' + '/*/*.npy')

    return sorted(glob.glob(f'/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/{dataset}/{feat}' + '/*/*.npy'))

def export_feature_list(feature_list, out_path):
    with open(out_path, 'w') as f:
        for path in feature_list:
            f.write(path.split('/')[-1].split('.')[-2] + '\t' + path + '\n')

def npy2h5py(feature_list_path, h5path, pca=None):
    paths = [l.split('\t')[1].strip() for l in open(feature_list_path, 'r').readlines()]

    with h5py.File(h5path, 'w') as f:
        for path in tqdm(paths):
            vid = path.split('/')[-1].split('.')[-2]
            if pca:
                f.create_dataset(vid, data=pca.infer(np.load_path))
            else:
                f.create_dataset(vid, data=np.load(path))   

if __name__ == "__main__":
    print("??")
    feature_list = get_feature_list(dataset='fivr', feat='avg')
    print("헤헤")
    export_feature_list(feature_list, out_path='fivr_feature_paths_mobilenetV2-avg.txt')

    feature_path = '/mldisk/nfs_shared_/js/contrastive_learning/fivr_feature_paths_mobilenetV2-avg.txt'
    paths = [l.split('\t')[1].strip() for l in open(feature_path, 'r').readlines()]

    # pool = Pool(32)
    # features = []
    # for path in paths:
    #     features += [pool.apply_async(pipe,
    #                                   args=[path],
    #                                   callback=(lambda *a: progress_bar.update()))]
    # pool.close()
    # pool.join()

    # feat_array = []
    # for feat in tqdm(features):
    #     feat_array.append(feat.get())
    # feats = np.concatenate(feat_array)
    #
    # pca = PCA(parameters_path='pca_params_vcdb997090_resnet50_imac_3840.npz')
    # pca.train(feats)