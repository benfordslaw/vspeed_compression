# vspeed compression

various methods to compress a video based on the speed of things moving in it. underlying logic: fast things are more important.

## usage

### `rip.py`

reduce each frame to only things that are moving fast enough. condense into the middle of the frame.

`python rip.py -i INPUT_FILEPATH`

### `combine_resized_fill.py`

normalize the video's speed by making faster things smaller than slower things AND fill in the gaps with nearby color values

`python combine_resized_fill.py -i INPUT_FILEPATH`

## dependencies
* opencv
* tqdm
* numpy
* ffmpeg
