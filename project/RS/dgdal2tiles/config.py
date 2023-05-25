

resampling_list = ('average', 'near', 'bilinear', 'cubic',
                   'cubicspline', 'lanczos', 'antialias')
profile_list = ('mercator', 'geodetic', 'raster')
webviewer_list = ('all', 'google', 'openlayers', 'leaflet', 'none')

MAXZOOMLEVEL = 32


DEFAULT_GDAL2TILES_OPTIONS = {
    'verbose': False,
    'title': '',
    'profile': 'mercator',
    'url': '',
    'resampling': 'average',
    's_srs': None,
    'zoom': None,
    'resume': False,
    'srcnodata': None,
    'tmscompatible': None,
    'quiet': False,
    'kml': False,
    'webviewer': 'all',
    'copyright': '',
    'googlekey': 'INSERT_YOUR_KEY_HERE',
    'bingkey': 'INSERT_YOUR_KEY_HERE',
    'nb_processes': 1
}

