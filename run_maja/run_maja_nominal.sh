#!/bin/sh
#$1 = tile_id
MAJA=/home/547/ea6141/datafile/opt/maja-3/bin/maja
INPUTS=maja_inputs/$1
OUTPUTS=maja_results/$1
CONF=Start_maja/userconf/
NTHREAD=16
L1DATA=s2_l1_data/

rm -fr $INPUTS
mkdir $INPUTS

regtime=$(find $OUTPUTS -name '*.EEF' | grep -oP '\d{8}')

cd $INPUTS
ln -s ../../maja_gip/* .
ln -s ../../S2__TEST_AUX_REFDE2_T${1}_0001/* .
ln -s ../../$OUTPUTS/* .
cd ../../

find $L1DATA -name "*${1}*"  |  grep -v $regtime | grep -v 'gml' | grep -v 'jp2' | grep -v 'xml' | grep -v 'gfs' | grep 'GRANULE' | \
    while read granule
    do
            parent=$(dirname $granule|xargs dirname)
            cd $INPUTS
            ln -s ../../$parent .
            cd ../../
            $MAJA -i $INPUTS -o $OUTPUTS --TileId $1 -ucs $CONF -m L2NOMINAL --NbThreads $NTHREAD
            rm $INPUTS/$(basename $parent)
    done


