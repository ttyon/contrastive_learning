import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from nets.models import *
import gc

import numpy as np
from PIL import Image
from tqdm import tqdm


class VCDBFrames(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/frames/vcdb/'):
        # vcdb core와 distraction 모두 한 폴더안에 모든 프레임이 들어가있다.
        self.paths = glob.glob(root + 'core/*.npy')
        self.paths += glob.glob(root + 'distraction/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]


class FIVRFrames(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/frames/fivr/'):
        self.paths = glob.glob(root + '*/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]


class CCWEBFrames(Dataset):
    def __init__(self, transform, root='~/datasets/CC_WEB_VIDEO/'):
        self.paths = glob.glob(root + 'Frames/*/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]


class EVVEFrames(Dataset):
    def __init__(self, transform, root='~/datasets/evve/'):
        self.paths = glob.glob(root + 'frames/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-1]


class Video2Vec():
    def __init__(self, dataset='vcdb', model='resnet-50', layers=['layer1', 'layer2', 'layer3', 'layer4'], feat='avg',
                 num_workers=4):
        """ Video2Vec
        :param model: String name of requested model
        :param layer: List of strings depending on model.
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.model_name = model
        self.feat = feat

        self.model, self.extraction_layers = self._get_model_and_layers(model, layers)

        self.model = self.model.cuda()

        self.model.eval()

        self.transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if dataset == 'vcdb':
            self.data_loader = DataLoader(VCDBFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'ccweb':
            self.data_loader = DataLoader(CCWEBFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'fivr':
            self.data_loader = DataLoader(FIVRFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'evve':
            self.data_loader = DataLoader(EVVEFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        else:
            raise KeyError('Dataset %s was not found' % dataset)

        print("len :", len(self.data_loader))

    def get_vec(self, path):
        """ Get vector embedding from a video(frames)
        :returns: Numpy ndarray
        """

        buffer = []

        # hook function
        def imac(module, input, output):
            logits = F.normalize(output.detach())
            logits = F.max_pool2d(logits, kernel_size=logits.size()[2:])
            logits = F.normalize(logits)
            buffer.append(logits.detach().cpu().numpy())

        def rmac(module, input, output):
            logits = F.normalize(output.detach())
            p = int(min(logits.size()[-2], logits.size()[-1]) * 2 / 7)  # 28->8 14->4 7->2
            logits = F.max_pool2d(logits, kernel_size=(int(p + p / 2), int(p + p / 2)), stride=p)  # (n, c, 3, 3)
            logits = logits.view(logits.size()[0], logits.size()[1], -1)  # (n, c, 9)
            logits = F.normalize(logits)
            logits = torch.sum(logits, dim=-1)
            logits = F.normalize(logits)
            buffer.append(logits.detach().cpu().numpy())

        def avg(module, input, output):
            logits = F.normalize(output.detach())
            logits = torch.nn.AdaptiveAvgPool2d((1, 1))(logits).squeeze(-1).squeeze(-1)
            logits = F.normalize(logits)
            buffer.append(logits.detach().cpu().numpy())

        with torch.no_grad():
            for data, suffix in tqdm(self.data_loader):
                # print("data :", data.shape)
                # print("suffix :", suffix)

                data = data[0]  # batch_size == 1, current shape (frames, c, h, w)
                suffix = self.feat + '/' + suffix[0]
                features = []

                if data.shape[0] == 0 or os.path.exists(path + 'features/vcdb' + suffix):
                    continue
                if data.shape[0] >= 600:
                    datas = torch.split(data, data.shape[0] // 2, dim=0)
                else:
                    datas = [data]

                for data in datas:
                    dt = self.model(data.cuda()).detach().cpu().numpy()
                    features.append(dt.tolist())

                # cuda memory batch size가 너무 크거나 코드 상에서 메모리 누수가 발생했기 때문에 일어남
                # gpu test 돌아가고 있어서 gc collect torch.cuda.empy_cache를 비워줘서 문제 해결
                # 대신 조금 오래 걸린다

                gc.collect()
                torch.cuda.empty_cache()
                np.save(path + 'features/vcdb/' + suffix, np.squeeze(np.concatenate(features, axis=0)))

                # features = []
                # for data in datas:
                #     buffer = []
                #     handles = []
                #     data = data.cuda()
                #     for layer in self.extraction_layers:
                #         if self.feat == 'imac':j
                #             h = layer.register_forward_hook(imac)
                #         elif self.feat == 'rmac':
                #             h = layer.register_forward_hook(rmac)
                #         elif self.feat == 'avg':
                #             h = layer.register_forward_hook(avg)
                #         handles.append(h)
                #     h_x = self.model(data)
                #     for h in handles:
                #         h.remove()
                #     del h_x
                #     del data
                #     features.append(np.concatenate(buffer, axis=1))
                # np.save(path + 'features/' + suffix, np.squeeze(np.concatenate(features, axis=0)))

    def _get_model_and_layers(self, model_name, layers):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-50'
        :param layers: layers as a string for resnet-50 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-50':
            model = models.resnet50(pretrained=True)
            model_layers = []
            for layer in layers:
                model_layers.append(model._modules.get(layer))
            print("model :", model)
            # print("model_layers :", model_layers)
            return model, model_layers
        elif model_name == 'mobilenet':
            model = MobileNet_AVG()
            return model, None
        else:
            raise KeyError('Model %s was not found' % model_name)


if __name__ == "__main__":
    print("???")
    v2v = Video2Vec(dataset='vcdb', num_workers=8)
    v2v.get_vec(path='/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/')
    # v2v.get_vec(path='/mldisk/nfs_shared_/js/contrastive_learning/pre_processing/features/fivr/rmac/features/rmac/core/')
    # / mldisk / nfs_shared_ / js / contrastive_learning / pre_processing / features / fivr / rmac / features / rmac / core /