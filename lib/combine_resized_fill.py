import os
import cv2 as cv
from numpy.lib.function_base import average
import random
import numpy as np
from tqdm import tqdm
import math

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Path to input video filename", type = str)
parser.add_argument("-fd", "--framedist", help="Distance between frames", default = 1, type = int)
parser.add_argument("-ds", "--downscale", help="Downscaling factor for faster pixels", default = 0.025, type = float)
parser.add_argument("-min", "--minmag", help="Lower threshold for window speed shown", default = 0.0, type = float)
parser.add_argument("-max", "--maxmag", help="Upper threshold for window speed shown", default = 1.0, type = float)
parser.add_argument("-p", "--padding", help="Padding to be added on each side of video", default = 25, type = int)
parser.add_argument("-o", "--output", help="Output directory to dump .png", default = "output", type = str)
parser.add_argument("-nf", "--no-fill", default=False, action="store_true", help="Turn off filling gaps")

args = parser.parse_args()

filename = args.input
frame_dist = args.framedist
ds_factor = args.downscale
min_mag_r = args.minmag
max_mag_r = args.maxmag
pad = args.padding
output = args.output

def init_normalize(total_frames, cap, sample_size, thres):
    accum_mag = []

    for _ in tqdm(range(sample_size)):

        f_ct = random.randint(0, total_frames-frame_dist-1)

        cap.set(1, f_ct)
        _, f1 = cap.read()
        cap.set(1, f_ct + frame_dist)
        _, f2 = cap.read()

        prvs = cv.cvtColor(f1, cv.COLOR_BGR2GRAY) 
        next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

        mag = get_mag(prvs, next, thres)

        accum_mag.extend(list(map(int, mag.flatten())))

    return(np.max(accum_mag))


def get_mag(prvs, next, thres):
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, thres, 3, 5, 1.2, 0)
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

    return mag

def main():
    cap = cv.VideoCapture(cv.samples.findFile(filename))
    cap_w = cap.get(3)
    cap_h = cap.get(4)
    min_dim = min(cap_w, cap_h)
    cap_w *= 200 / min_dim
    cap_h *= 200 / min_dim
    cap_w = int(cap_w)
    cap_h = int(cap_h)
    thres = 10 
    total = int(cap.get(7))
    title_txt = filename.split("file_in/")[1]

    _, fast_img = cap.read()
    fast_img = cv.resize(fast_img, (cap_w, cap_h), cv.INTER_NEAREST)
    prvs = cv.cvtColor(fast_img, cv.COLOR_BGR2GRAY)

    max_mag = init_normalize(total, cap, 100, thres)
    min_mag_allowed = max_mag * min_mag_r 
    max_mag_allowed = max_mag * max_mag_r
    (txt_width, txt_height), baseline = cv.getTextSize(title_txt, cv.FONT_HERSHEY_PLAIN, 1, 1)

    if not os.path.exists(output):
        os.makedirs(output)

    for frame_ct in tqdm(np.arange(0, total-frame_dist, frame_dist)):
        accum_img = np.zeros_like(fast_img)

        cap.set(1, frame_ct + frame_dist)
        _, f2 = cap.read()
        f2 = cv.resize(f2, (cap_w, cap_h), cv.INTER_NEAREST)

        next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

        mag = get_mag(prvs, next, thres)

        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                if mag[i][j] > min_mag_allowed and mag[i][j] < max_mag_allowed:
                    h_ds = (ds_factor * mag.shape[0] * mag[i][j] / max_mag) + 1
                    v_ds = (ds_factor * mag.shape[1] * mag[i][j] / max_mag) + 1
                    h_idx = math.floor(i/h_ds + mag.shape[0] * (h_ds - 1) / (2 * h_ds))
                    v_idx = math.floor(j/v_ds + mag.shape[1] * (v_ds - 1) / (2 * v_ds))
                    accum_img[h_idx][v_idx] = f2[i][j]

        if not args.no_fill:
            c_acc_i = int(mag.shape[0] / 2)
            c_acc_j = int(mag.shape[1] / 2)
            for i in range(1, mag.shape[0]-1):
                for j in range(1, mag.shape[1]-1):
                    if accum_img[i][j].sum() == 0:
                        angle_to_center = math.atan2(j - c_acc_j, i - c_acc_i)
                        i_step = math.cos(angle_to_center)
                        j_step = math.sin(angle_to_center)
                        min_step = min(abs(i_step), abs(j_step))
                        if min_step != 0:
                            i_step /= min_step
                            j_step /= min_step
                        i_step = int(i_step)
                        j_step = int(j_step)
                        i_c = i
                        j_c = j
                        while j_c + j_step < mag.shape[1] and i_c + i_step < mag.shape[0] and j_c + j_step > 0 and i_c + i_step > 0:
                            i_c += i_step
                            j_c += j_step
                            if accum_img[i_c][j_c].sum() != 0:
                                accum_img[i][j] = accum_img[i_c][j_c]
                                break
                        
        pad_img = cv.copyMakeBorder(accum_img, pad, pad, pad, pad, cv.BORDER_CONSTANT)
        txt_img = cv.putText(pad_img, title_txt, (pad, pad + mag.shape[0] + txt_height+baseline), cv.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv.LINE_8)
        cv.imwrite(output + "/" + str(thres) + "_" + str('{:0>6}'.format(frame_ct)) + ".png", txt_img)

        prvs = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

if __name__ == "__main__":
    main()
