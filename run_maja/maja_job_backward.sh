#!/bin/bash
JOBDIR=$PWD
NCPUS=16
MEM=128GB
JOBFS=10GB

cat $1 | \
    while read granule
    do
        qsub -P u46 -q normal -l walltime=48:00:00,mem=$MEM,jobfs=$JOBFS,ncpus=$NCPUS,wd -N maja_$granule -- \
    bash -l -c "\
    source $HOME/.bashrc;\
    module load openmpi/3.0.0; cd $JOBDIR; \
    ./run_maja_backward.sh $granule; wait"
    done

