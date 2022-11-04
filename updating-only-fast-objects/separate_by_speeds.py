import sys
sys.path.append('/Users/benkrummenacher/opt/anaconda3/envs/vpype_env/lib/python3.10/site-packages')

import cv2 as cv
from numpy.lib.function_base import average
import random
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

thres = int(sys.argv[2])
# mag_thresh = int(sys.argv[2])
frame_dist = int(sys.argv[3])

def takeSecond(elem):
    return elem[1]

def init_normalize(total_frames, cap, sample_size):

    max_mag = 0

    accum_mag = []

    for _ in tqdm(range(sample_size)):

        f_ct = random.randint(0, total_frames-frame_dist-1)

        cap.set(1, f_ct)
        _, f1 = cap.read()

        prvs = cv.cvtColor(f1, cv.COLOR_BGR2GRAY) 
        
        cap.set(1, f_ct + frame_dist)
        _, f2 = cap.read()

        next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

        mag = get_mag(prvs, next)

        accum_mag.extend(list(map(int, mag.flatten())))

    # plt.hist(accum_mag, facecolor='g', bins=int(np.max(accum_mag)), histtype='step')
    # plt.axvline(x = np.mean(accum_mag), color = 'r')
    # plt.show()
    mean_thresh = np.mean(accum_mag)

    return(mean_thresh)

    # return max_mag, mag, accum_mag


def get_mag(prvs, next):
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, thres, 3, 5, 1.2, 0)
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

    return mag

def main():

    cap = cv.VideoCapture(cv.samples.findFile(sys.argv[1]))
    total = int(cap.get(7))

    _, fast_img = cap.read()
    prvs = cv.cvtColor(fast_img, cv.COLOR_BGR2GRAY)

    mag_thresh = init_normalize(total, cap, 50)
    
    num_seps = int(sys.argv[4])

    next_updates = []
    for k in range(num_seps):
        next_updates.append(np.zeros_like(fast_img))

    accum_img = fast_img

    for frame_ct in tqdm(np.arange(0, total-1, frame_dist)):

        cap.set(1, frame_ct)
        _, frame2 = cap.read()

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        norm_mag = get_mag(prvs, next)

        for k in np.arange(num_seps, 0, -1):
            if frame_ct % (k*frame_dist) == 0:
                for l in range(next_updates[k-1].shape[0]):
                    for m in range(next_updates[k-1].shape[1]):
                        if next_updates[k-1][l][m].sum() > 0:
                            accum_img[l][m] = next_updates[k-1][l][m]
                            next_updates[k-1][l][m] = [0,0,0]
                for i in range(norm_mag.shape[0]):
                    for j in range(norm_mag.shape[1]):
                        if norm_mag[i, j] >= 2*(k - 1)/num_seps * mag_thresh and (norm_mag[i, j] < 2*k/num_seps * mag_thresh or k==num_seps):
                            next_updates[k-1][i, j] = frame2[i, j]

        cv.imwrite("output/" + str(thres) + "_" + str('{:0>4}'.format(frame_ct)) + ".png", accum_img)
        # comb_img = np.concatenate(seps, axis=1)

        # cv.imwrite("output/" + str(thres) + "_" + str('{:0>4}'.format(frame_ct)) + ".png", comb_img)

        prvs = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

if __name__ == "__main__":
    main()
