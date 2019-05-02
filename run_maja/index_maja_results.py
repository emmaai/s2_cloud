import datacube
import sys
import rasterio
import numpy as np
import time
import os
import subprocess
from datacube.storage import netcdf_writer
from datacube.storage.storage import create_netcdf_storage_unit, write_dataset_to_netcdf
import xarray as xr
from os import path
from datacube.model import Measurement  
from datacube.model import DatasetType as Product
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.utils.geometry import box
from pathlib import Path
from datacube.api.query import query_group_by, query_geopolygon
import yaml
from yaml import CSafeLoader as Loader, CSafeDumper as Dumper
import re
import uuid

def _create_product(metadata_type, product_name, product_type, data_measurements, storage):
    product_definition = {
    'name': product_name,
    'description': 'Description for ' + product_name,
    'metadata_type': metadata_type.name,
    'metadata': {
    'product_type': product_type,
    'format': {'name': 'GeoTIFF'},
    'instrument': {'name': 'TM,ETM+,OLI'},
    'platform': {'code': 'SENTINEL_2'}
    },
    'storage': storage,
    'measurements': data_measurements
    }
    Product.validate(product_definition)
    return Product(metadata_type, product_definition)

def load_data(x1, x2, y1, y2, start, end, product = 's2a_level1c_granule'):
    dc = datacube.Datacube()
    # Set up spatial and temporal query; note that 'output_crs' and 'resolution' need to be set
    query = {'x' : (x1, x2),
         'y': (y1, y2),
         'time': (start, end),
         'crs': 'EPSG:3577',
         'output_crs': 'EPSG:3577'
         }   
    
    datasets = dc.find_datasets(product = product,  group_by='solar_day', 
                          **query) 
    return datasets

def main(x1, x2, y1, y2, start, end):
    product_name = 'maja_cloud_geo_mask'
    product_type = 'maja_cloud_geo_mask'
    datasets = load_data(x1, x2, y1, y2, start, end)
    measurements = []
    for name in ['cloud_mask', 'geo_mask', 'data_mask']:
        measurements += [Measurement(name=name, dtype='int16', nodata=-1, units='1')]

    results_dir = '/g/data/u46/users/ea6141/s2_cloud/run_maja/'
    for dataset in datasets:
        file_path = dataset.measurements['B01']['path']
        r = re.compile(r"(?<=_T)\d{2}\w{3}")
        tile_id = np.unique(r.findall(file_path))
        time_stamp = dataset.metadata_doc['extent']['center_dt']
        time_stamp = time_stamp.replace('-', '').replace('T', '-').replace(':', '').replace('.', '-').strip('Z')
        if os.path.exists('maja_results/' + tile_id[0]) == False:
            continue
        for item in os.listdir('maja_results/' + tile_id[0]):
            if time_stamp in item:
                path_product = os.path.join('maja_results/' + tile_id[0],  item)
                path_cloud = os.path.join(path_product+'/MASKS', item+'_CLM_R1.tif')
                path_geo = os.path.join(path_product+'/MASKS', item+'_MG2_R1.tif')
                path_mask = os.path.join(path_product+'/MASKS', item+'_EDG_R1.tif')

                if os.path.exists(path_geo) == False:
                    continue
                storage = {'crs': dataset.crs.wkt,
                           'resolution':{'x': 10, 'y': -10},
                           'tile_size': {'x': dataset.bounds.width, 'y': dataset.bounds.height}
                           }
                product = _create_product(dataset.metadata_type, product_name, product_type, measurements, storage)
                extent = box(dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top, dataset.crs)
                output_dataset = make_dataset(product, [dataset], extent, dataset.center_time, valid_data=dataset.extent)
                #output_dataset.metadata_doc['lineage'] = {'source_datasets': {}}
                output_dataset = xr.DataArray([output_dataset], coords = [[dataset.center_time]],  dims=['time'])
                output_dataset = datasets_to_doc(output_dataset)
                dataset_object = (output_dataset.item()).decode('utf-8')
                yaml_obj = yaml.load(dataset_object, Loader=Loader)
                yaml_obj['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, results_dir+path_product))
                
                for key, value in yaml_obj['image']['bands'].items():
                    value['layer'] = str(1)
                    if key == 'cloud_mask':
                        value['path'] = results_dir + path_cloud
                    elif key == 'geo_mask':
                        value['path'] = results_dir + path_geo
                    else:
                        value['path'] = results_dir + path_mask
                yaml_fname = 'maja_yaml/' + item + '.yaml'
                with open(yaml_fname, 'w') as fp:
                    yaml.dump(yaml_obj, fp, default_flow_style=False, Dumper=Dumper)
        
if __name__ == '__main__':

    nargv = len(sys.argv)
    if nargv!=7:
        print("Usage: index_maja_results.py x1 x2 y1 y2 start end")
        exit()
    elif nargv==7:
        args = sys.argv[1:]
    main(*args)
