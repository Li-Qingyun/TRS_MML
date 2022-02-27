import scipy.io as sio
import torch
import numpy as np
from torch.utils.data import TensorDataset


if __name__ == '__main__':

    nTrain = 200
    nValidate = 50
    dataset_name = 'Pavia'  # TODO

    # TODO
    # hsi_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\Indian_pines_corrected.mat"
    # gt_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\Indian_pines_gt.mat"
    # hsi_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\Kennedy_denoise.mat"
    # gt_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\KSC_gt.mat"
    # hsi = sio.loadmat(hsi_path)['Kennedy176']
    # gt = sio.loadmat(gt_path)['KSC_gt']
    hsi_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\Pavia.mat"
    gt_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\Pavia_groundtruth.mat"
    hsi = sio.loadmat(hsi_path)['Pavia']
    gt = sio.loadmat(gt_path)['groundtruth']
    # hsi_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\Indian_pines_corrected.mat"
    # gt_path = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\Indian_pines_gt.mat"
    save_dir = r"C:\Users\LQY\Desktop\TRS-MML\data\datasets\hsi_cls\\"

    hsi = 1 * ((hsi - np.min(hsi)) / (np.max(hsi) - np.min(hsi)) - 0.5)
    hsi = hsi.transpose([2, 0, 1])

    nBand, nRow, nCol = hsi.shape
    nClass = int(np.max(gt))

    HalfWidth = 16
    windowsize = 2 * HalfWidth + 1
    NotZeroMask = np.zeros([nRow, nCol])
    Wid = 2 * HalfWidth
    NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
    gt = gt * NotZeroMask
    [nnRow, nnCol] = np.nonzero(gt)
    nSample = np.size(nnRow)

    gt = gt - 1

    nTest = nSample - nTrain - nValidate

    RandPerm = np.random.permutation(nSample)
    sio.savemat(save_dir + '{}_row_col.mat'.format(dataset_name), {'row': nnRow, 'col': nnCol, 'RandPerm': RandPerm})

    trainX = np.zeros([nTrain, nBand, windowsize, windowsize], dtype=np.float32)
    trainY = np.zeros([nTrain], dtype=np.int64)
    ValX = np.zeros([nValidate, nBand, windowsize, windowsize], dtype=np.float32)
    ValY = np.zeros([nValidate], dtype=np.int64)
    testX = np.zeros([nTest, nBand, windowsize, windowsize], dtype=np.float32)
    testY = np.zeros([nTest], dtype=np.int64)
    for i in range(nTrain):
        trainX[i, :, :, :] = hsi[:, nnRow[RandPerm[i]] - HalfWidth: nnRow[RandPerm[i]] + HalfWidth + 1, \
                             nnCol[RandPerm[i]] - HalfWidth: nnCol[RandPerm[i]] + HalfWidth + 1]
        trainY[i] = gt[nnRow[RandPerm[i]], nnCol[RandPerm[i]]].astype(np.int64)

    for i in range(nValidate):
        ValX[i, :, :, :] = hsi[:, nnRow[RandPerm[i + nTrain]] - HalfWidth: nnRow[RandPerm[i + nTrain]] + HalfWidth + 1, \
                           nnCol[RandPerm[i + nTrain]] - HalfWidth: nnCol[RandPerm[i + nTrain]] + HalfWidth + 1]
        ValY[i] = gt[nnRow[RandPerm[i + nTrain]], nnCol[RandPerm[i + nTrain]]].astype(np.int64)

    for i in range(nTest):
        testX[i, :, :, :] = hsi[:, nnRow[RandPerm[i + nTrain + nValidate]] - HalfWidth: nnRow[RandPerm[
            i + nTrain + nValidate]] + HalfWidth + 1, \
                            nnCol[RandPerm[i + nTrain + nValidate]] - HalfWidth: nnCol[RandPerm[
                                i + nTrain + nValidate]] + HalfWidth + 1]
        testY[i] = gt[nnRow[RandPerm[i + nTrain + nValidate]], nnCol[RandPerm[i + nTrain + nValidate]]].astype(
            np.int64)

    np.save(save_dir + '{}_corrected_train'.format(dataset_name), trainX)
    np.save(save_dir + '{}_corrected_test'.format(dataset_name), testX)
    np.save(save_dir + '{}_gt_train'.format(dataset_name), trainY)
    np.save(save_dir + '{}_gt_test'.format(dataset_name), testY)