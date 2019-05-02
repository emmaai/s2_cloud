import datacube
import sys
import numpy as np
from os import path
import re
from zipfile import ZipFile

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


def get_granules(datasets, output_folder):
    granules = []
    for dataset in datasets:
        file_path = dataset.measurements['B01']['path']
        zipfile = file_path.split('!')[0].lstrip('zip:')
        datafile = file_path.split('!')[1].split('/')[0]
        r = re.compile(r"(?<=_T)\d{2}\w{3}")
        granules += r.findall(file_path)
        if path.exists(path.join(output_folder, datafile)):
            continue
        with ZipFile(zipfile, 'r') as zf:
            zf.extractall(output_folder)

    granules = np.unique(granules)
    return granules

def main(x1, x2, y1, y2, start, end):
    datasets = load_data(*args)
    granules = get_granules(datasets, 's2_l1_data')

    with open('_'.join(['s2_l1_granules', str(x1), str(x2), str(y1), str(y2), start, end]), 'w') as f:
        for g in granules:
            f.write(g+'\n')


if __name__ == '__main__':

    nargv = len(sys.argv)
    if nargv!=2 and nargv!=7:
        print("Usage: find_l1data.py x1 x2 y1 y2 start end")
        exit()

    if nargv==2:
        args = sys.argv[1].split()
    elif nargv==7:
        args = sys.argv[1:]
    main(*args)

    
