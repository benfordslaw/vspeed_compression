#!/bin/bash

# usage: bash tools/download_video.sh VIDEOSRC MAX_DIM

mkdir file_in/tmp
yt-dlp -o "file_in/tmp/%(title)s.%(ext)s" -f "worst[ext=mp4]" $1
filename=$(youtube-dl --get-filename -o "%(title)s.%(ext)s" $1)
max_dim=$2
ffmpeg -y -i "file_in/tmp/$filename" -vf "scale=min(iw*$max_dim/ih\,$max_dim):min($max_dim\,ih*$max_dim/iw)" "file_in/$filename"
rm -r file_in/tmp
