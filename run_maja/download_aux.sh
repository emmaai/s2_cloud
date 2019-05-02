#!/bin/bash

cat $1 | \
    while read tile_id
    do
        python3 Start_maja/prepare_mnt/tuilage_mnt_eau_S2.py -s $tile_id.txt -p parametres.txt -m SRTM -d
    done
