#!/bin/bash
cd ../build
DATASETS=(ak2010 asia_osm belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 delaunay_n24 hollywood-2009 kron_g500-logn21 road_central road_usa roadNet-CA soc-twitter-2010) 
P_VALUES=(0.1 0.3 0.5 0.7 0.9)
# distributed experiments over 3 gpus, those being 1, 2, 3 
I=0
for d in "${DATASETS[@]}"; do
    export CUDA_VISIBLE_DEVICES=$((I%3+1))
    nohup ./bin/tc -m ../datasets/${d}/${d}.mtx > ../experiments/res/${d}.txt &
    I=$((I+1))
    for p in "${P_VALUES[@]}"; do
        export CUDA_VISIBLE_DEVICES=$((I%3+1))
        p_int=$(echo "$p * 100 / 1" | bc)
        nohup ./bin/tc -m ../datasets/${d}/${d}.mtx -s $p > ../experiments/res/${d}-${p_int}.txt &
        I=$((I+1))
    done
done