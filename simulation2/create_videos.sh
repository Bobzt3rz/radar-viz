#!/bin/bash

pushd output
pushd projections
ffmpeg -y -framerate 20 -i frame_%08d_proj.png -c:v libopenh264 -b:v 10M -maxrate 10M -profile:v high -pix_fmt yuv420p a_proj.mp4
popd
pushd clustered_projections
ffmpeg -y -framerate 20 -i frame_%08d_proj.png -c:v libopenh264 -b:v 10M -maxrate 10M -profile:v high -pix_fmt yuv420p a_clust.mp4
popd
pushd filtered_projections
ffmpeg -y -framerate 20 -i frame_%08d_proj.png -c:v libopenh264 -b:v 10M -maxrate 10M -profile:v high -pix_fmt yuv420p a_filt.mp4
popd
popd
