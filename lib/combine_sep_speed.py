import sys
sys.path.append('/Users/benkrummenacher/opt/anaconda3/envs/vpype_env/lib/python3.10/site-packages')

import os
import cv2 as cv
from numpy.lib.function_base import average
import random
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
 
parser.add_argument("-i", "--input", help="Path to input video filename", type = str)
parser.add_argument("-w", "--winsize", help="Optical flow window size", default = 10, type = int)
parser.add_argument("-fd", "--framedist", help="Distance between frames", default = 1, type = int)
parser.add_argument("-d", "--depth", help="Depth of mag window analysis", default = 2, type = int)
parser.add_argument("-o", "--output", help="Output directory to dump .png", default = "output", type = str)

args = parser.parse_args()

filename = args.input
thres = args.winsize
frame_dist = args.framedist
depth = args.depth
output = args.output

def get_mean_mags(accum_means, mag_in):
    if len(accum_means) < 2^depth + 1:
        new_mag = []
        for mag in mag_in:
            mean = np.mean(mag)
            accum_means.extend([mean])
            mag_l = [m_el for m_el in mag if m_el < mean]
            mag_h = [m_el for m_el in mag if m_el >= mean]
            new_mag.extend([mag_l, mag_h])
        return get_mean_mags(accum_means, new_mag) 
    else:
        accum_means.sort()
        return accum_means

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

    accum_means = get_mean_mags([], [accum_mag])
    accum_means.insert(0, 0)
    accum_means.append(np.max(accum_mag))

    return(accum_means)


def get_mag(prvs, next):
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, thres, 3, 5, 1.2, 0)
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

    return mag

def main():
    cap = cv.VideoCapture(cv.samples.findFile(filename))
    total = int(cap.get(7))

    _, fast_img = cap.read()
    prvs = cv.cvtColor(fast_img, cv.COLOR_BGR2GRAY)

    accum_means = init_normalize(total, cap, 50)
    accum_img = fast_img

    num_seps = len(accum_means)-1

    if not os.path.exists(output):
        os.makedirs(output)

    for frame_ct in tqdm(np.arange(0, total-1, frame_dist)):
        for k in range(len(accum_means)-1):
            cap.set(1, (frame_ct * (num_seps - k))%total)
            _, f1 = cap.read()

            cap.set(1, (frame_ct * (num_seps - k) + frame_dist)%total)
            _, f2 = cap.read()

            prvs = cv.cvtColor(f1, cv.COLOR_BGR2GRAY) 
            next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

            mag = get_mag(prvs, next)

            for i in range(mag.shape[0]):
                for j in range(mag.shape[1]):
                    if mag[i][j] >= accum_means[k] and mag[i][j] < accum_means[k+1]:
                        accum_img[i][j] = f2[i][j]
    
        cv.imwrite(output + "/" + str(thres) + "_" + str('{:0>6}'.format(frame_ct)) + ".png", accum_img)

if __name__ == "__main__":
    main()
