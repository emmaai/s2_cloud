#!/bin/bash

SOURCE=s2_l1_data

cat $1 | \
    while read tile_id
    do
        xmlfile=$(find $SOURCE -name '*.xml' | grep $tile_id | grep 'GRANULE' | grep -v '\QI_DATA' | head -1)
        python3 sites_params.py $tile_id $xmlfile
    done
