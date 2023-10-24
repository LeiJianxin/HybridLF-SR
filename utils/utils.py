# from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
import h5py
from einops import rearrange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import metrics
import math


class CVSTrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(CVSTrainSetLoader, self).__init__()
        self.datasetDir = cfg.trainsetDir
        self.angRes = cfg.angRes
        self.inList = cfg.inList
        self.dataList = cfg.dataList
        self.fileList = []
        for dataName in self.dataList:
            tmpList = os.listdir(self.datasetDir + dataName)
            for index, _ in enumerate(tmpList):
                tmpList[index] = dataName + '/' + tmpList[index]
            self.fileList.extend(tmpList)
        self.itemNum = len(self.fileList)

    def __getitem__(self, index):
        fileName = [self.datasetDir + self.fileList[index % self.itemNum]]
        with h5py.File(fileName[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label_c'))

        data, label = augmentation(data, label)
        daH, daW = data.shape
        input_index = random.choice(self.inList)
        data = rearrange(data, '(u h) (v w) -> (u v) h w', u=self.angRes, h=daH // self.angRes, v=self.angRes,
                         w=daW // self.angRes)
        data = rearrange(data[input_index, :, :], '(u v) h w -> (u h) (v w)', u=2, v=2)
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.itemNum


class HLFSSRTrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(HLFSSRTrainSetLoader, self).__init__()
        self.datasetDir = cfg.trainsetDir
        self.dataList = cfg.dataList
        self.fileList = []
        for dataName in self.dataList:
            tmpList = os.listdir(self.datasetDir + dataName)
            for index, _ in enumerate(tmpList):
                tmpList[index] = dataName + '/' + tmpList[index]
            self.fileList.extend(tmpList)
        self.itemNum = len(self.fileList)

    def __getitem__(self, index):
        fileName = [self.datasetDir + self.fileList[index % self.itemNum]]
        with h5py.File(fileName[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label_SSR'))
            hr = np.array(hf.get('hr'))

        data, label, hr = augmentation3(data, label, hr)
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())
        hr = ToTensor()(hr.copy())

        return data, label, hr

    def __len__(self):
        return self.itemNum


class BDTrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(BDTrainSetLoader, self).__init__()
        self.datasetDir = cfg.trainsetDir
        self.dataList = cfg.dataList
        self.fileList = []
        for dataName in self.dataList:
            tmpList = os.listdir(self.datasetDir + dataName)
            for index, _ in enumerate(tmpList):
                tmpList[index] = dataName + '/' + tmpList[index]
            self.fileList.extend(tmpList)
        self.itemNum = len(self.fileList)

    def __getitem__(self, index):
        fileName = [self.datasetDir + self.fileList[index % self.itemNum]]
        with h5py.File(fileName[0], 'r') as hf:
            data = np.array(hf.get('hr'))
            label = np.array(hf.get('label_c'))

        data, label = augmentation(data, label)
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.itemNum


def MultiTestSetDataLoader(cfg):
    testLoaders = []

    for dataName in cfg.testList:
        if 'BD' in cfg.modelName:
            testDataset = BDTestSetDataLoader(cfg, dataName)
        if 'CVS' in cfg.modelName:
            testDataset = CVSTestSetDataLoader(cfg, dataName)
        if 'HLFSSR' in cfg.modelName:
            testDataset = HLFSSRTestSetDataLoader(cfg, dataName)

        testLoaders.append(DataLoader(dataset=testDataset, batch_size=1, shuffle=False))
    return testLoaders


class HLFSSRTestSetDataLoader(Dataset):
    def __init__(self, cfg, dataName='ALL'):
        super(HLFSSRTestSetDataLoader, self).__init__()

        self.datasetDir = cfg.testsetDir + dataName
        self.fileList = []
        tmpList = os.listdir(self.datasetDir)
        for index, _ in enumerate(tmpList):
            tmpList[index] = tmpList[index]
        self.fileList.extend(tmpList)
        self.itemNum = len(self.fileList)

    def __getitem__(self, index):
        fileName = self.datasetDir + '/' + self.fileList[index]
        with h5py.File(fileName, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label_SSR'))
            hr = np.array(hf.get('hr'))
        data, label, hr = np.transpose(data, (1, 0)), np.transpose(label, (1, 0)), np.transpose(hr, (1, 0))
        data, label, hr = ToTensor()(data.copy()), ToTensor()(label.copy()), ToTensor()(hr.copy())

        return data, label, hr

    def __len__(self):
        return self.itemNum


class CVSTestSetDataLoader(Dataset):
    def __init__(self, cfg, dataName='ALL'):
        super(CVSTestSetDataLoader, self).__init__()
        self.angRes = cfg.angRes
        self.inList = cfg.inList
        self.datasetDir = cfg.testsetDir + dataName
        self.fileList = []
        tmpList = os.listdir(self.datasetDir)
        for index, _ in enumerate(tmpList):
            for idxList in range(len(self.inList)):
                self.fileList.append(tmpList[index])
        self.itemNum = len(self.fileList)

    def __getitem__(self, index):
        fileName = self.datasetDir + '/' + self.fileList[index]
        with h5py.File(fileName, 'r') as hf:
            data = np.array(hf.get('label_SSR'))
            label = np.array(hf.get('hr'))

        daH, daW = data.shape
        inIndex = self.inList[index % len(self.inList)]
        data = rearrange(data, '(u h) (v w) -> (u v) h w', u=self.angRes, h=daH // self.angRes, v=self.angRes, w=daW // self.angRes)
        data = rearrange(data[inIndex, :, :], '(u v) h w -> (u h) (v w)', u=2, v=2)
        data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
        data, label = ToTensor()(data.copy()), ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.itemNum


class BDTestSetDataLoader(Dataset):
    def __init__(self, cfg, dataName='ALL'):
        super(BDTestSetDataLoader, self).__init__()
        self.datasetDir = cfg.testsetDir + dataName
        self.fileList = []
        tmpList = os.listdir(self.datasetDir)
        for index, _ in enumerate(tmpList):
            tmpList[index] = tmpList[index]

        self.fileList.extend(tmpList)
        self.itemNum = len(self.fileList)

    def __getitem__(self, index):
        fileName = self.datasetDir + '/' + self.fileList[index]
        with h5py.File(fileName, 'r') as hf:
            data = np.array(hf.get('hr'))
            label = np.array(hf.get('label_c'))
        data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
        data, label = ToTensor()(data.copy()), ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.itemNum


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


def augmentation3(data, label, hr):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
        hr = hr[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
        hr = hr[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
        hr = hr.transpose(1, 0)
    return data, label, hr


def LF2SAIs(lf):
    N, an2, h, w = lf.shape
    an = int(math.sqrt(an2))
    SAIs = rearrange(lf.view(N, an, an, h, w), 'b u v h w -> b (u h) (v w)').view(N, 1, an * h, an * w)
    return SAIs


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np, data_range=1)


def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, data_range=1)


def cal_metrics(img1, img2, angRes):
    if len(img1.size()) == 2:
        [H, W] = img1.size()
        img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0, 2, 1, 3)
    if len(img2.size()) == 2:
        [H, W] = img2.size()
        img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0, 2, 1, 3)

    [U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')

    for u in range(U):
        for v in range(V):
            if angRes > 2 and u == U // 2 and v == V // 2:
                PSNR[u, v] = 0.0
                SSIM[u, v] = 0.0
            else:
                PSNR[u, v] = metrics.peak_signal_noise_ratio(img1[u, v, :, :].data.cpu().numpy(), img2[u, v, :, :].data.cpu().numpy())
                SSIM[u, v] = metrics.structural_similarity(img1[u, v, :, :].data.cpu().numpy(), img2[u, v, :, :].data.cpu().numpy(), gaussian_weights=True)
            pass
        pass

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean


def cal_metrics_HR(img1, img2):
    PSNR = cal_psnr(img1, img2)
    SSIM = cal_ssim(img1, img2)
    return PSNR, SSIM


def compt_psnr(img1, img2, angRes):
    [N, _, aH, aW] = img1.size()

    img1 = img1.view(N, angRes, aH // angRes, angRes, aW // angRes).permute(0, 1, 3, 2, 4)
    img2 = img2.view(N, angRes, aH // angRes, angRes, aW // angRes).permute(0, 1, 3, 2, 4)

    [N, U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(N, U, V), dtype='float32')
    for n in range(N):
        for u in range(U):
            for v in range(V):
                if angRes > 2 and u == U // 2 and v == V // 2:
                    PSNR[n, u, v] = 0.0
                else:
                    PSNR[n, u, v] = cal_psnr(img1[n, u, v, :, :], img2[n, u, v, :, :])
                pass
            pass
    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    return psnr_mean


def compt_psnr_lr(img1, img2):
    [N, _, H, W] = img1.size()
    PSNR = []
    for n in range(N):
        PSNR.append(cal_psnr(img1[n, 0, :, :], img2[n, 0, :, :]))
    psnr_mean = np.array(PSNR).mean()
    return psnr_mean


def cal_metrics_c(img1, img2):
    img1 = img1.squeeze(0).squeeze(0)
    img2 = img2.squeeze(0).squeeze(0)
    PSNR = cal_psnr(img1, img2)
    SSIM = cal_ssim(img1, img2)

    return PSNR, SSIM


def gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


def lf2epi(lf):
    N, an, an, h, w = lf.shape
    epi_h_wo_c = []
    epi_v_wo_c = []
    epi_h = lf.permute(0, 1, 3, 2, 4).contiguous().view(-1, 1, an, w)
    epi_v = lf.permute(0, 2, 4, 1, 3).contiguous().view(-1, 1, an, h)
    for i in range(an):
        if i == an // 2:
            continue
        epi_h_wo_c.append(epi_h[i::an])
        epi_v_wo_c.append(epi_v[i::an])
    epi_h_wo_c = torch.cat(epi_h_wo_c, 0)
    epi_v_wo_c = torch.cat(epi_v_wo_c, 0)
    return epi_h_wo_c, epi_v_wo_c


def SAIs2LF(SAIs, anRes):
    N, _, ah, aw = SAIs.shape
    h = ah // anRes
    w = aw // anRes
    lf = rearrange(SAIs.view(N, ah, aw), 'b (u h) (v w) -> b u v h w', b=N, u=anRes, h=h, v=anRes, w=w)
    return lf
