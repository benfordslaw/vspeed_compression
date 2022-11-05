import os
import cv2 as cv
from numpy.lib.function_base import average
import random
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Path to input video filename", type = str, required = True)
parser.add_argument("-w", "--winsize", help="Optical flow window size", default = 10, type = int)
parser.add_argument("-fd", "--framedist", help="Distance between frames", default = 1, type = int)
parser.add_argument("-min", "--minmag", help="Minimum speed for pixels to update, out of 255", default = 50, type = int)
parser.add_argument("-o", "--output", help="Output directory to dump .png", default = "output", type = str)

args = parser.parse_args()

filename = args.input
thres = args.winsize
frame_dist = args.framedist
mag_thresh = args.minmag
output = args.output

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

    cap = cv.VideoCapture(cv.samples.findFile(filename))
    total = int(cap.get(7))

    _, acc_img_rgb = cap.read()
    prvs = cv.cvtColor(acc_img_rgb, cv.COLOR_BGR2GRAY)

    (max_mag, prv_norm_mag) = init_normalize(total, cap, 50)
    max_mag /= 255
    prv_norm_mag /= max_mag

    if not os.path.exists(output):
        os.makedirs(output)

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

        cv.imwrite(output + "/" + str(thres) + "_" + str('{:0>4}'.format(frame_ct)) + ".png", acc_img_rgb)

        prvs = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        prv_norm_mag = norm_mag

if __name__ == "__main__":
    main()
