# vspeed compression

various methods to compress a video based on the speed of things moving in it. 

## usage

### rip

reduce each frame to only things that are moving fast enough. condense into the middle of the frame.

`python lib/rip.py -i INPUT_FILEPATH`

### combine resized, fill

normalize the video's speed by making faster things smaller than slower things AND fill in the gaps with nearby color values

`python lib/combine_resized_fill.py -i INPUT_FILEPATH`

### download videos

download and resize a youtube video given a url and maximum dimension

`bash tools/download_video.sh "URL" MAX_DIM`

## dependencies
* opencv
* tqdm
* numpy
* ffmpeg
* yt-dlp
