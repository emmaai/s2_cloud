
import sys, os
import numpy as np
import xarray as xr
import time
import pcm

from datacube.drivers.netcdf import netcdf_writer
from datacube.drivers.netcdf import create_netcdf_storage_unit, write_dataset_to_netcdf
import xarray as xr
from os import path
from datacube.model import Measurement  
from datacube.model import DatasetType as Product
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from pathlib import Path
from datacube.api.query import query_group_by, query_geopolygon

from datacube import Datacube
import s2cloudmask as s2cm
import pickle

def _create_product(version, metadata_type, product_name, product_type, data_measurements, storage):
    product_definition = {
    'name': product_name,
    'description': 'Description for ' + product_name,
    'metadata_type': metadata_type.name,
    'metadata': {
    'product_type': product_type+'_'+version,
    'format': {'name': 'NetCDF'},
    'instrument': {'name': 'TM,ETM+,OLI'},
    'platform': {'code': 'SENT2'}
    },
    'storage': storage,
    'measurements': data_measurements
    }
    Product.validate(product_definition)
    return Product(metadata_type, product_definition)

def _write_dataset_to_netcdf(dataset, filename, output_dataset, global_attributes=None, variable_params=None,
                                    netcdfparams=None):
    """
    Write a Data Cube style xarray Dataset to a NetCDF file

    Requires a spatial Dataset, with attached coordinates and global crs attribute.

    :param `xarray.Dataset` dataset:
    :param filename: Output filename
    :param global_attributes: Global file attributes. dict of attr_name: attr_value
    :param variable_params: dict of variable_name: {param_name: param_value, [...]}
    Allows setting storage and compression options per variable.
    See the `netCDF4.Dataset.createVariable` for available
    parameters.
    :param netcdfparams: Optional params affecting netCDF file creation
    """
    global_attributes = global_attributes or {}
    variable_params = variable_params or {}
    filename = Path(filename)

    if not dataset.data_vars.keys():
        raise DatacubeException('Cannot save empty dataset to disk.')

    if not hasattr(dataset, 'crs'):
        raise DatacubeException('Dataset does not contain CRS, cannot write to NetCDF file.')


    nco = create_netcdf_storage_unit(filename,
                                    dataset.crs,
                                    dataset.coords,
                                    dataset.data_vars,
                                    variable_params,
                                    global_attributes,
                                    netcdfparams)
    output_dataset.source = None
    output_dataset = xr.DataArray([output_dataset], coords = [dataset.time.values],  dims=['time'])
    output_dataset = datasets_to_doc(output_dataset)

    netcdf_writer.create_variable(nco, 'dataset', output_dataset, zlib=True)
    nco['dataset'][:] = netcdf_writer.netcdfy_data(output_dataset.values)
    for name, variable in dataset.data_vars.items():
        nco[name][:] = netcdf_writer.netcdfy_data(variable.values)

    nco.close()

def _find_datasets(query, dc, product = 's2a_level1c_granule'):
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    datasets = dc.find_datasets(product = product, measurements = bands, group_by='solar_day', 
                          **query) 
    group_by_tile = {}
    for d in datasets:
        if group_by_tile.get(d.metadata_doc['image']['tile_reference'], None) is None:
            group_by_tile[d.metadata_doc['image']['tile_reference']] = [d]
        else:
            group_by_tile[d.metadata_doc['image']['tile_reference']].append(d)
    return group_by_tile

def load_and_mask(query, dc, group_by_tile, dir_name, fout, product = 's2a_level1c_granule'):
    
    __version__ = 'v1.0'
    product_name = 's2_cloud_mask_spectral'
    product_type = 's2_cloud_mask_spectral'
    storage = {'crs': 'EPSG:3577',
               'resolution':{'x': 10, 'y': -10},
               'tile_size': {'x': 100000.0, 'y': 100000.0}
               }

    metadata_type = dc.index.metadata_types.get_by_name('eo')
    measurements = [Measurement(name='cloud_mask', dtype='int16', nodata=-1, units='1')]
    spectral_product = _create_product(__version__, metadata_type, product_name, product_type, measurements, storage)

    reference = dict() 
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    for key, datasets in group_by_tile.items():
        gm_pkl = 'gm_' + key + '.pkl'
        if path.exists(gm_pkl) == False:
            obs_stack = [] 
            for d in datasets:
                sentinel_ds = dc.load(product = product, datasets=[d], measurements = bands, group_by='solar_day', **query)
                sentinel_ds = sentinel_ds.to_array(dim='bands')
                sentinel_ds = sentinel_ds.transpose(*(sentinel_ds.dims[1:]+sentinel_ds.dims[:1]))
                obs, cloud_mask = s2_cloud_mask(sentinel_ds, spectral_product, d, dir_name, fout+'_'+key)
                obs_stack.append(obs)
                clear_obs = obs.copy()
                clear_obs[cloud_mask == 1] = -1
                if reference.get(key, None) is None:
                    reference[key] = [clear_obs]
                else:
                    reference[key].append(clear_obs)
            reference[key] = np.transpose(np.array(reference[key], dtype=np.float32), [1,2,3,0])
            obs_stack = np.transpose(np.array(obs_stack, dtype=np.float32), [1, 2, 3, 0])
            reference[key][(reference[key] < 0).all(axis=(2, 3))] = obs_stack[(reference[key] < 0).all(axis=(2, 3))]
            reference[key][reference[key] < 0] = np.nan
            reference[key] = pcm.gmpcm(reference[key], num_threads=7)
            with open(gm_pkl, 'wb') as f:
                pickle.dump(reference[key], f)
        else:
            with open(gm_pkl, 'rb') as f:
                reference[key] = pickle.load(f)
    return reference

def s2_cloud_mask(data_array, product, dataset, dir_name, fout):
    obs = data_array[0].data.astype('float32')
    output_fname = path.join(dir_name, fout+'_'+str(data_array.time.values[0]).replace(':', '_')+'.nc')
    try:
        obs[obs > 10000] = 10000 
        obs[obs < 1] = np.nan
    except:
        print('do nothing')
    obs /= 10000
    if path.exists(output_fname):
        print('read in cloud mask file', output_fname)
        cloud_mask = xr.open_dataset(output_fname)
        cloud_mask = cloud_mask.cloud_mask_spectral
    else:
        cloud_mask = s2cm.cloud_mask(obs, model='spectral').astype('int16')
        cloud_mask[np.any(np.isnan(obs), axis=2)] = -1
        cloud_mask = xr.Dataset({"cloud_mask":(['time', 'y', 'x'], cloud_mask.reshape((1, )+cloud_mask.shape))}, 
            coords={'time':data_array.time.values, 'y': data_array.y, 'x':data_array.x},
            attrs={'crs': data_array.attrs['crs']})
        cloud_mask.coords['time'].attrs = data_array.time.attrs
        print(output_fname)
        output_dataset = make_dataset(product, [dataset], dataset.extent, cloud_mask.time.values[0], uri=Path(output_fname).as_uri())
        _write_dataset_to_netcdf(cloud_mask, output_fname, output_dataset)
        cloud_mask = cloud_mask.cloud_mask

    return (obs, cloud_mask)


def load_and_shadow_mask(query, dc, group_by_tile, dir_name, fout, reference, product = 's2a_level1c_granule'):
    __version__ = 'v1.0'
    product_name = 's2_cloud_shadow_mask_temporal'
    product_type = 's2_cloud_shadow_mask_temporal'
    storage = {'crs': 'EPSG:3577',
               'resolution':{'x': 10, 'y': -10},
               'tile_size': {'x': 100000.0, 'y': 100000.0}
               }

    metadata_type = dc.index.metadata_types.get_by_name('eo')
    measurements = [Measurement(name='cloud_shadow_mask', dtype='int16', nodata=-1, units='1')]
    temporal_product = _create_product(__version__, metadata_type, product_name, product_type, measurements, storage)

    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    for key, datasets in group_by_tile.items():
        for d in datasets:
            sentinel_ds = dc.load(product = product, datasets=[d], measurements = bands, group_by='solar_day', **query)
            sentinel_ds = sentinel_ds.to_array(dim='bands')
            sentinel_ds = sentinel_ds.transpose(*(sentinel_ds.dims[1:]+sentinel_ds.dims[:1]))
            s2_shadow_mask(sentinel_ds, reference[key], temporal_product, d, dir_name, fout+'_' + key)

def s2_shadow_mask(data_array, reference, product, dataset, dir_name, fout):
    output_fname = path.join(dir_name, fout+'_'+str(data_array.time.values[0]).replace(':', '_')+'.nc')
    if path.exists(output_fname):
        return
    print(output_fname)
    obs = data_array[0].data.astype('float32')
    try:
        obs[obs > 10000] = 10000 
        obs[obs < 1] = np.nan
    except:
        print('do nothing')
    obs /= 10000
    shadow_mask = s2cm.shadow_mask(obs, model='fast-shadow', ref=reference)
    cloud_mask = s2cm.cloud_mask(obs, model='temporal', ref=reference)
    cloud_shadow_mask = np.zeros(cloud_mask.shape, dtype='int16')
    cloud_shadow_mask[cloud_mask] = 1
    cloud_shadow_mask[shadow_mask] += 2
    cloud_shadow_mask[np.any(np.isnan(obs), axis=2)] = -1
    cloud_shadow_mask = xr.Dataset({"cloud_shadow_mask":(['time', 'y', 'x'], 
                                cloud_shadow_mask.reshape((1, )+cloud_shadow_mask.shape).astype('int16'))}, 
                                coords={'time':data_array.time.values, 'y': data_array.y, 'x':data_array.x},
                                attrs={'crs': data_array.attrs['crs']})
    cloud_shadow_mask.coords['time'].attrs = data_array.time.attrs
    output_fname = path.join(dir_name, fout+'_'+str(data_array.time.values[0]).replace(':', '_')+'.nc')
    output_dataset = make_dataset(product, [dataset], dataset.extent, data_array.time.values[0], uri=Path(output_fname).as_uri())
    #output_dataset.metadata_doc['lineage'] = {'source_datasets': {}}

    _write_dataset_to_netcdf(cloud_shadow_mask, output_fname, output_dataset)


def main(x1, x2, y1, y2, start, end):

    __version__ = 'v1.0'
    
    dc = Datacube()
    # Set up spatial and temporal query; note that 'output_crs' and 'resolution' need to be set
    query = {'x' : (x1, x2),
         'y': (y1, y2),
         'time': (start, end),
         'crs': 'EPSG:3577',
         'output_crs': 'EPSG:3577',
         'resolution': (10, -10)}   
    
    dataset_pkl = '_'.join(['tile', str(x1), str(x2), str(y1), str(y2), str(start), str(end)]) + '.pkl'
    if path.exists(dataset_pkl) == False:
        datasets = _find_datasets(query, dc)
        with open(dataset_pkl, 'wb') as f:
            pickle.dump(datasets, f)
    else:
        with open(dataset_pkl, 'rb') as f:
            datasets = pickle.load(f)

    dir_name = '_'.join([__version__, 'tile', str(x1), str(x2), str(y1), str(y2), str(start), str(end)])
    os.makedirs(dir_name, exist_ok=True)
    dir_name = path.join(os.environ['PWD'], dir_name) 

    
    
    fout = '_'.join(['cloud_mask_spectral', str(x1), str(x2), str(y1), str(y2)])
    reference = load_and_mask(query, dc, datasets, dir_name, fout)

   
    fout = '_'.join(['cloud_shadow_mask_temporal', str(x1), str(x2), str(y1), str(y2)])
    load_and_shadow_mask(query, dc, datasets, dir_name, fout, reference)

#argstr = "1150000 1200000 -4250000 -4200000 2016-07-01 2016-08-01"
if __name__ == '__main__':

    nargv = len(sys.argv)
    if nargv!=2 and nargv!=7:
        print("Usage: make_s2cm.py x1 x2 y1 y2 start end")
        exit()

    if nargv==2:
        args = sys.argv[1].split()
    elif nargv==7:
        args = sys.argv[1:]
    main(*args)
