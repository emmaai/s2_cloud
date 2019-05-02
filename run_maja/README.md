# run_maja
Run MAJA standalone over sentinel 2
===================================

It was intended to produce cloud mask for Sentinel 2. The process is broken down into 3 parts:
1. query the Sentinel 2 level 1 data with the `datacube-core`
2. generate necessary granule parameters required by the `Start_maja/prepare_mnt/tuilage_mnt_eau_S2.py`, which is a tool to convert auxiliary data into the format required by *MAJA*

    Note: it needs particularly this one https://github.com/emmaai/Start_maja

3. gather all the inputs and invoke *MAJA*

Usage
-----
1. `python find_l1data.py x1 x2 y1 y2 start_date end_date`
  find the l1 granules tile id and unzip the l1 data files as the inputs of *MAJA*
      
      outputs:
      `$granule_id_file`
      
2. `./generate_sites_params $granule_id_file`
  generate the inputs for the tools of converting auxiliary data
  
    outputs:
      `$granule_id.txt`
      
3. `./download_convert_aux.sh $granule_id_file`
  find what the auxiliary data will be needed, if they're not available, dump a list of files; if they're availabe, convert into the form *MAJA* requires
 
4. `./run_maja_backward.sh $granule_id`
  run *MAJA* in BACKWARD_CHAIN with a stack of input files of `$granule_id`. L2 output of this step will be used as reference in producing the rest L2 results
  
5. `./run_maja_nominal.sh $granule_id`
  run *MAJA* in NOMINAL_CHAIN for each L1 input of `$granule_id`
  
6. `python index_maja_results.py x1 x2 y1 y2 start_date end_date`
  index `cloud mask` and `geographical mask` produced by *MAJA* into database so that they can be queried and loaded by `datacube`
