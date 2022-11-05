# vspeed compression

various methods to compress a video based on the speed of things moving in it. 

## usage

### rip

reduce each frame to only things that are moving fast enough. condense into the middle of the frame.

`rip.py [-h] [-i INPUT] [-fd FRAMEDIST] [-min MINMAG] [-max MAXMAG] [-p PADDING]`

### combine resized, fill

normalize the video's speed by making faster things smaller than slower things AND fill in the gaps with nearby color values

`combine_resized_fill.py [-h] [-i INPUT] [-fd FRAMEDIST] [-ds DOWNSCALE] [-min MINMAG] [-max MAXMAG] [-p PADDING] [-nf NO_FILL]`

### combine, separated by speed

normalize the video's speed by making the frame rate of faster things smaller than that of slower things

`combine_sep_speed.py [-h] [-i INPUT] [-w WINSIZE] [-fd FRAMEDIST] [-d DEPTH]`

### download videos

download and resize a youtube video given a url and maximum dimension

`bash tools/download_video.sh "URL" MAX_DIM`

## dependencies
* opencv
* tqdm
* numpy
* ffmpeg
* yt-dlp
