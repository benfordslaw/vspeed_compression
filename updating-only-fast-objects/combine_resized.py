import sys
sys.path.append('/Users/benkrummenacher/opt/anaconda3/envs/vpype_env/lib/python3.10/site-packages')

import cv2 as cv
from numpy.lib.function_base import average
import random
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

thres = int(sys.argv[2])
frame_dist = int(sys.argv[3])
ds_factor = float(sys.argv[4])
min_mag_r = float(sys.argv[5])
max_mag_r = float(sys.argv[6])

def takeSecond(elem):
    return elem[1]

def init_normalize(total_frames, cap, sample_size):
    accum_mag = []

    for _ in tqdm(range(sample_size)):

        f_ct = random.randint(0, total_frames-frame_dist-1)

        cap.set(1, f_ct)
        _, f1 = cap.read()
        cap.set(1, f_ct + frame_dist)
        _, f2 = cap.read()

        prvs = cv.cvtColor(f1, cv.COLOR_BGR2GRAY) 
        next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

        mag = get_mag(prvs, next)

        accum_mag.extend(list(map(int, mag.flatten())))
        
    return(np.max(accum_mag))


def get_mag(prvs, next):
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, thres, 3, 5, 1.2, 0)
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

    return mag

def main():
    cap = cv.VideoCapture(cv.samples.findFile(sys.argv[1]))
    total = int(cap.get(7))

    _, fast_img = cap.read()
    prvs = cv.cvtColor(fast_img, cv.COLOR_BGR2GRAY)

    max_mag = init_normalize(total, cap, 100)

    for frame_ct in tqdm(np.arange(0, total-1, frame_dist)):
        accum_img = np.zeros_like(fast_img)

        cap.set(1, frame_ct + frame_dist)
        _, f2 = cap.read()

        next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

        mag = get_mag(prvs, next)
        min_mag_allowed = max_mag * min_mag_r 
        max_mag_allowed = max_mag * max_mag_r

        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                if mag[i][j] > min_mag_allowed and mag[i][j] < max_mag_allowed:
                    h_ds = (ds_factor * mag.shape[0] * mag[i][j] / max_mag) + 1
                    v_ds = (ds_factor * mag.shape[1] * mag[i][j] / max_mag) + 1
                    h_idx = math.floor(i/h_ds + mag.shape[0] * (h_ds - 1) / (2 * h_ds))
                    v_idx = math.floor(j/v_ds + mag.shape[1] * (v_ds - 1) / (2 * v_ds))
                    accum_img[h_idx][v_idx] = f2[i][j]
                    #accum_img[h_idx][v_idx] = [255, 255, 255]
    
        cv.imwrite("output/" + str(thres) + "_" + str('{:0>6}'.format(frame_ct)) + ".png", accum_img)

        prvs = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

if __name__ == "__main__":
    main()
