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
parser.add_argument("-min", "--minmag", help="Lower threshold for window speed shown", default = 0.0, type = float)
parser.add_argument("-max", "--maxmag", help="Upper threshold for window speed shown", default = 1.0, type = float)
parser.add_argument("-p", "--padding", help="Padding to be added on each side of video", default = 25, type = int)
parser.add_argument("-o", "--output", help="Output directory to dump .png", default = "output", type = str)

args = parser.parse_args()

filename = args.input
frame_dist = args.framedist
min_mag_r = args.minmag
max_mag_r = args.maxmag
pad = args.padding 
output = args.output

def init_normalize(total_frames, cap, sample_size, thres):
    print("initializing speed normalization value")
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
    cap_w = int(cap_w)
    cap_h = int(cap_h)
    total = int(cap.get(7))
    thres = 10
    title_txt = filename

    _, fast_img = cap.read()
    prvs = cv.cvtColor(fast_img, cv.COLOR_BGR2GRAY)

    max_mag = init_normalize(total, cap, 100, thres)
    min_mag_allowed = max_mag * min_mag_r 
    max_mag_allowed = max_mag * max_mag_r
    title_txt = str(min_mag_allowed) + " - " + str(max_mag_allowed)
    (txt_width, txt_height), baseline = cv.getTextSize(title_txt, cv.FONT_HERSHEY_PLAIN, 1, 1)

    if not os.path.exists(output):
        os.makedirs(output)

    print("ripping frames")

    for frame_ct in tqdm(np.arange(0, total-frame_dist, frame_dist)):
        accum_img = np.zeros_like(fast_img)

        cap.set(1, frame_ct + frame_dist)
        _, f2 = cap.read()

        next = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

        mag = get_mag(prvs, next, thres)

        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                if mag[i][j] > min_mag_allowed and mag[i][j] < max_mag_allowed:
                    accum_img[i][j] = f2[i][j]

        h_condensed = np.zeros_like(accum_img)
        indices = np.where(np.all(accum_img != (0,0,0), axis=-1))
        indexes = zip(indices[0], indices[1])
        i_s = 0
        j_s = 0
        vert_c = int(h_condensed.shape[1] / 2)
        for (p0, p1) in indexes:
            horiz_line_length = np.count_nonzero(indices[0] == p0)
            h_condensed[i_s][j_s + vert_c - int(horiz_line_length / 2)] = accum_img[p0][p1]
            j_s += 1
            if j_s == horiz_line_length:
                j_s = 0
                i_s += 1

        v_condensed = np.zeros_like(accum_img)
        indices = np.where(np.all(h_condensed != (0,0,0), axis=-1))
        indexes = sorted(zip(indices[1], indices[0]))
        i_s = 0
        j_s = 0
        horiz_c = int(v_condensed.shape[0] / 2)
        for (p0, p1) in indexes:
            vert_line_length = np.count_nonzero(indices[1] == p0)
            v_condensed[i_s + horiz_c - int(vert_line_length / 2)][p0] = h_condensed[p1][p0]
            i_s += 1
            if i_s == vert_line_length:
                i_s = 0
                j_s += 1

        pad_img = cv.copyMakeBorder(v_condensed, pad, pad, pad, pad, cv.BORDER_CONSTANT)
        #txt_img = cv.putText(pad_img, title_txt, (pad, pad + mag.shape[0] + txt_height+baseline), cv.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv.LINE_8)
        cv.imwrite(output + "/" + str('{:0>6}'.format(frame_ct)) + ".png", pad_img)

        prvs = cv.cvtColor(f2, cv.COLOR_BGR2GRAY) 

if __name__ == "__main__":
    main()
