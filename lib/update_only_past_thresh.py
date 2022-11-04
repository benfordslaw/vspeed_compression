import sys
sys.path.append('/Users/benkrummenacher/opt/anaconda3/envs/vpype_env/lib/python3.10/site-packages')

import cv2 as cv
from numpy.lib.function_base import average
import random
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

thres = int(sys.argv[3])
mag_thresh = int(sys.argv[2])
frame_dist = int(sys.argv[4])

def takeSecond(elem):
    return elem[1]

def init_normalize(total_frames, cap, sample_size):

    max_mag = 0

    for _ in tqdm(range(sample_size)):
        random_frame_ct = random.randint(0, total_frames-frame_dist-1)
        cap.set(1, random_frame_ct)
        _, f1 = cap.read()

        prvs = cv.cvtColor(f1, cv.COLOR_BGR2GRAY) 
        
        cap.set(1, random_frame_ct + frame_dist)
        _, f2 = cap.read()

        next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

        mag = get_mag(prvs, next)

        if np.max(mag) > max_mag:
            max_mag = np.max(mag)

    return max_mag, mag


def get_mag(prvs, next):
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, thres, 3, 5, 1.2, 0)
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

    return mag

def main():

    cap = cv.VideoCapture(cv.samples.findFile(sys.argv[1]))
    total = int(cap.get(7))

    _, acc_img_rgb = cap.read()
    prvs = cv.cvtColor(acc_img_rgb, cv.COLOR_BGR2GRAY)

    (max_mag, prv_norm_mag) = init_normalize(total, cap, 50)
    max_mag /= 255
    prv_norm_mag /= max_mag

    for frame_ct in tqdm(np.arange(0, total-1, frame_dist)):

        cap.set(1, frame_ct)
        _, frame2 = cap.read()

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        norm_mag = get_mag(prvs, next) / max_mag
    
        for i in range(norm_mag.shape[0]):
            for j in range(norm_mag.shape[1]):
                if norm_mag[i, j] >= mag_thresh:
                    acc_img_rgb[i, j] = frame2[i, j]
                elif prv_norm_mag[i, j] >= mag_thresh:
                    acc_img_rgb[i, j] = frame2[i, j]

        cv.imwrite("output/" + str(thres) + "_" + str('{:0>4}'.format(frame_ct)) + ".png", acc_img_rgb)

        prvs = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        prv_norm_mag = norm_mag

if __name__ == "__main__":
    main()
