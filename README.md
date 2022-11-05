# vspeed compression

various methods to compress a video based on the speed of things moving in it. 

## usage

mkdir `vspeed_compression/output/`, required for outputting .png

### rip

reduce each frame to only things that are moving fast enough. condense into the middle of the frame.

`rip.py [-h] [-i INPUT] [-fd FRAMEDIST] [-min MINMAG] [-max MAXMAG] [-p PADDING] [-o OUTPUT]`

### combine resized, fill

normalize the video's speed by making faster things smaller than slower things AND fill in the gaps with nearby color values

`combine_resized_fill.py [-h] [-i INPUT] [-fd FRAMEDIST] [-ds DOWNSCALE] [-min MINMAG] [-max MAXMAG] [-p PADDING] [-o OUTPUT] [-nf]`

### combine, separated by speed

normalize the video's speed by making the frame rate of faster things smaller than that of slower things

`combine_sep_speed.py [-h] [-i INPUT] [-w WINSIZE] [-fd FRAMEDIST] [-d DEPTH] [-o OUTPUT]`

### update only past threshold

only update parts of a video that are moving faster than a given threshold (out of 255, default 50)

`update_only_past_thresh.py [-h] [-i INPUT] [-w WINSIZE] [-fd FRAMEDIST] [-min MINMAG] [-o OUTPUT]`

### download videos

download and resize a youtube video given a url and maximum dimension

`bash tools/download_video.sh "URL" MAX_DIM`

## dependencies
* opencv
* tqdm
* numpy
* ffmpeg
* yt-dlp
