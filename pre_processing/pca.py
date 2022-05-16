import os
from multiprocessing import Pool
# from progress_bar import InitBar?
import progress_bar
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from pre_processing.feature_post_processing import get_feature_list, npy2h5py, export_feature_list


class PCA():
    def __init__(self, n_components=1024, whitening=True,
                 parameters_path='models/pca_params_vcdb997090_resnet50_rmac_3840.npz'):
        self.n_components = n_components
        self.whitening = whitening
        self.parameters_path = parameters_path

    def train(self, x):
        '''training pca.
        Args:
            x: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        '''

        # x = x.t()
        x = torch.Tensor(x).t() # 나는 tensor로 저장 안 했옹
        nPoints = x.size(1)
        nDims = x.size(0)

        # x = x.double()
        mu = x.mean(1).unsqueeze(1)
        x = x - mu

        if (nDims <= nPoints):
            doDual = False
            x2 = torch.matmul(x, x.t()) / (nPoints - 1)
        else:
            doDual = True
            x2 = torch.matmul(x.t(), x) / (nPoints - 1)

        L, U = torch.symeig(x2, eigenvectors=True)
        if (self.n_components < x2.size(0)):
            k_indices = torch.argsort(L, descending=True)[:self.n_components]
            L = torch.index_select(L, 0, k_indices)
            U = torch.index_select(U, 1, k_indices)

        lams = L
        lams[lams < 1e-9] = 1e-9

        if (doDual):
            U = torch.matmul(x, torch.matmul(U, torch.diag(1. / torch.sqrt(lams)) / np.sqrt(nPoints - 1)))

        Utmu = torch.matmul(U.t(), mu)

        U, lams, mu, Utmu = U.numpy(), lams.numpy(), mu.numpy(), Utmu.numpy()

        print('================= PCA RESULT ==================')
        print('U: {}'.format(U.shape))
        print('lams: {}'.format(lams.shape))
        print('mu: {}'.format(mu.shape))
        print('Utmu: {}'.format(Utmu.shape))
        print('===============================================')

        # save features, labels to h5 file.
        filename = os.path.join(self.parameters_path)
        np.savez(filename, U=U, lams=lams, mu=mu, Utmu=Utmu)

    def load(self):
        print('loading PCA parameters...')
        pca = np.load(self.parameters_path)
        U = pca['U'][...][:, :self.n_components]
        lams = pca['lams'][...][:self.n_components]
        mu = pca['mu'][...]
        Utmu = pca['Utmu'][...]

        if (self.whitening):
            U = np.matmul(U, np.diag(1./np.sqrt(lams)))
        Utmu = np.matmul(U.T, mu)

        self.weight = torch.from_numpy(U.T).view(self.n_components, -1, 1, 1).float()
        self.bias = torch.from_numpy(-Utmu).view(-1).float()

    def infer(self, data):
        '''apply PCA/Whitening to data.
        Args:
            data: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        Returns:
            output: [N, output_dim] FloatTensor with output of PCA/Whitening operation.
        '''

        N, D = data.size()
        data = data.view(N, D, 1, 1)
        if torch.cuda.is_available():
            output = F.conv2d(data, self.weight.cuda(), bias=self.bias.cuda(), stride=1, padding=0).view(N, -1)
        else:
            output = F.conv2d(data, self.weight, bias=self.bias, stride=1, padding=0).view(N, -1)

        output = F.normalize(output, p=2, dim=-1) # IMPORTANT!
        assert (output.size(1) == self.n_components)
        return output

def pipe(a):
    a = np.load(a)
    a = a[np.random.choice(len(a), 10), :]
    return a

def test():
    print("None")

# train gpu 설정 방법
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

feature_list = get_feature_list(dataset='vcdb', feat='avg')
export_feature_list(feature_list, out_path='/mldisk/nfs_shared_/js/contrastive_learning/vcdb_feature_paths_mobilenet_avg_1280.txt')

## PCA whitening
print("PCA whitening")
feature_path='/mldisk/nfs_shared_/js/contrastive_learning/vcdb_feature_paths_mobilenet_avg_1280.txt'
paths = [l.split('\t')[1].strip() for l in open(feature_path, 'r').readlines()]

features = []
pool = Pool(32)
for path in tqdm(paths):
    features += [pipe(path)]
pool.close()
pool.join()

feat_array = []
for feat in tqdm(features):
    feat_array.append(feat)
feats = np.concatenate(feat_array)

pca = PCA(parameters_path='/workspace/pre_processing/pca_params_vcdb_mobilenet_avg_1280.npz')
pca.train(feats)

## pca applying
print("pca applying")
pca.load()

def f(path):
    feat = np.load(path)
    feat = pca.infer(torch.Tensor(feat).cuda()).cpu().numpy()
    path = path.replace('avg', 'avg_pca1024')

    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(path, feat)

paths = get_feature_list(dataset='vcdb', feat='avg')

for path in tqdm(paths):
    if not os.path.exists(path.replace('avg', 'avg_pca1024')):
        f(path)

## final step
print("final step")
feature_list = get_feature_list(dataset='vcdb', feat='avg_pca1024')
print(len(feature_list))
export_feature_list(feature_list, out_path='/mldisk/nfs_shared_/js/contrastive_learning/vcdb_feature_paths_mobilenet_avg_pca1024.txt')

feature_path = '/mldisk/nfs_shared_/js/contrastive_learning/vcdb_feature_paths_mobilenet_avg_pca1024.txt'
h5path = 'vcdb_mobilenet_avg_pca1024.hdf5'
npy2h5py(feature_path, h5path, pca=None)

## final step
# print("final step")
# feature_list = get_feature_list(dataset='fivr', feat='avg_pca1024')
# export_feature_list(feature_list, out_path='/mldisk/nfs_shared_/js/contrastive_learning/fivr_feature_paths_mobilenet_avg_pca1024.txt')
# feature_path = '/mldisk/nfs_shared_/js/contrastive_learning/vcdb_feature_paths_resnet50_imac_pca1024.txt'
# h5path = 'fivr_mobilenet_avg_pca1024.hdf5'
# npy2h5py(feature_path, h5path, pca=None)

# feature_list = get_feature_list(dataset='fivr', feat='imac_pca1024')
# export_feature_list(feature_list, out_path='/mldisk/nfs_shared_/js/contrastive_learning/fivr_feature_paths_resnet50_imac_pca1024.txt')
# feature_path = '/mldisk/nfs_shared_/js/contrastive_learning/vcdb_feature_paths_resnet50_imac_pca1024.txt'
# h5path = 'fivr_resnet_imac_pca1024.hdf5'
# npy2h5py(feature_path, h5path, pca=None)
#
# feature_list = get_feature_list(dataset='fivr', feat='imac_pca1024')
# export_feature_list(feature_list, out_path='/mldisk/nfs_shared_/js/contrastive_learning/fivr_feature_paths_resnet50_rmac_pca1024.txt')
# feature_path = '/mldisk/nfs_shared_/js/contrastive_learning/fivr_feature_paths_resnet50_rmac_3840.txt'
# h5path = 'fivr_resnet_rmac_pca1024.hdf5'
# npy2h5py(feature_path, h5path, pca=None)