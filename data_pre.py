import numpy as np
from torch.utils.data import Dataset
import torch
import os
from skimage import io
# from sklearn.model_selection import train_test_split
# import random

MEAN_OF_I = 2.5006557136946483
STD_OF_I = 16.761658638753033

MEAN_OF_Q = 10.451615952346184
STD_OF_Q = 25.64762332975976

# MEAN_OF_IR = 166.33927640874336
# STD_OF_IR = 57.18249332488938

# MEAN_OF_IRequ = 156.53883958748892
# STD_OF_IRequ = 62.65655712315825

# MEAN_OF_depth = 100.86308666472739
# STD_OF_depth = 82.40279578758052

# random.seed(0)


def getFrameId(path):
    files = [f for f in os.listdir(path) if f.endswith('txt')]
    frames = [int(name.split('_')[1]) for name in files]
    frames = list(np.unique(np.array(frames)))
    return frames

def hist_equ(im):
    imhist, bins = np.histogram(im.flatten(), 256)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    im2 = im2.reshape(im.shape)
    # cv.imwrite('tmp2.png',im2)
    return im2
    
class IQ_dataset(Dataset):
    def __init__(self, iq_dir, target_dir, depth_dir):
        self.iq_dir = iq_dir
        self.target_dir = target_dir
        self.depth_dir = depth_dir
        self.frames = getFrameId(iq_dir)
        # print(self.frames)

    def __len__(self):
        return int(len(self.frames))

    def __getitem__(self, idx):

        frame_id = self.frames[idx]
        ii = np.loadtxt(os.path.join(self.iq_dir, 'frame_{:d}_1.txt'.format(frame_id)))
        iq = np.loadtxt(os.path.join(self.iq_dir, 'frame_{:d}_2.txt'.format(frame_id)))

        amp = np.sqrt(np.power(ii,2) + np.power(iq,2))
        phase = np.arctan2(iq,ii)
        phase[phase<0] += 2*np.pi 

        N2 = amp * phase / (2 * np.pi)
        N1 = amp - N2

        beta = np.arctan(np.power(N2, 2) / (N1 + N2 + 1e-1))

        ii = (ii - MEAN_OF_I) / STD_OF_I
        iq = (iq - MEAN_OF_Q) / STD_OF_Q

        item = torch.from_numpy(np.array([ii, iq]).astype(np.float32))

        # print("IQ shape: ",item.shape)#IQ shape:  torch.Size([2, 480, 640])

        target = (beta - beta.min())/(beta.max() - beta.min())
        target = torch.from_numpy(target.astype(np.float32))

        # print("IR shape: ",target.shape)#IR shape:  torch.Size([480, 640])
        # target = target.view(1, target.shape[0], target.shape[1])

        depth = phase / (2*np.pi)
        depth = torch.from_numpy(depth.astype(np.float32))
        
        # print("depth shape: ",depth.shape)#depth shape:  torch.Size([480, 640])

        return item, target, depth, frame_id
        

