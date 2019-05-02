import xml.etree.ElementTree as ET
import sys

def main(tile_id, xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    utmzone = root[1][0][0].text
    crs = root[1][0][1].text
    ulx = root[1][0][5][0].text
    uly = root[1][0][5][1].text
    with open(tile_id+'.txt', 'w') as f:
        f.write('proj=UTM' + ''.join(list(utmzone)[-3:]) + '\n')
        f.write('EPSG_out=' + crs.split(':')[1] + '\n')
        f.write('chaine_proj=' + crs + '\n')
        f.write('tx_min=0\n')
        f.write('ty_min=0\n')
        f.write('tx_max=0\n')
        f.write('ty_max=0\n')
        f.write('pas_x=109800\n')
        f.write('pas_y=109800\n')
        f.write('orig_x=' + ulx + '\n')
        f.write('orig_y=' + uly + '\n')
        f.write('marge=0\n')


if __name__ == '__main__':

    nargv = len(sys.argv)
    if nargv!=3:
        print("Usage: sites_params.py tile_id xmlfile")
        exit()

    args = sys.argv[1:]
    main(*args)

    
