# -*- coding: utf-8 -*-
import os
import numpy
import warnings
from osgeo import gdal

from .config import DEFAULT_GDAL2TILES_OPTIONS
from .error_info import exit_with_error


class AttrDict(object):
    """
    Helper class to provide attribute like access (read and write) to
    dictionaries. Used to provide a convenient way to access both results and
    nested dsl dicts.
    """
    def __init__(self, d={}):
        # assign the inner dict manually to prevent __setattr__ from firing
        super(AttrDict, self).__setattr__('_d_', d)

    def __contains__(self, key):
        return key in self._d_

    def __nonzero__(self):
        return bool(self._d_)
    __bool__ = __nonzero__

    def __dir__(self):
        # introspection for auto-complete in IPython etc
        return list(self._d_.keys())

    def __eq__(self, other):
        if isinstance(other, AttrDict):
            return other._d_ == self._d_
        # make sure we still equal to a dict with the same data
        return other == self._d_

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        r = repr(self._d_)
        if len(r) > 60:
            r = r[:60] + '...}'
        return r

    def __getstate__(self):
        return (self._d_, )

    def __setstate__(self, state):
        super(AttrDict, self).__setattr__('_d_', state[0])

    def __getattr__(self, attr_name):
        try:
            return self.__getitem__(attr_name)
        except KeyError:
            raise AttributeError(
                '%r object has no attribute %r' % (self.__class__.__name__, attr_name))

    def __delattr__(self, attr_name):
        try:
            del self._d_[attr_name]
        except KeyError:
            raise AttributeError(
                '%r object has no attribute %r' % (self.__class__.__name__, attr_name))

    def __getitem__(self, key):
        return self._d_[key]

    def __setitem__(self, key, value):
        self._d_[key] = value

    def __delitem__(self, key):
        del self._d_[key]

    def __setattr__(self, name, value):
        if name in self._d_ or not hasattr(self.__class__, name):
            self._d_[name] = value
        else:
            # there is an attribute on the class (could be property, ..) - don't add it as field
            super(AttrDict, self).__setattr__(name, value)

    def __iter__(self):
        return iter(self._d_)

    def to_dict(self):
        return self._d_


def recursive_attrdict(obj):
    """
    .. deprecated:: version

    Walks a simple data structure, converting dictionary to AttrDict.
    Supports lists, tuples, and dictionaries.
    """
    warnings.warn("deprecated", DeprecationWarning)
    AttrDict(obj)

def process_options(input_file, output_folder, options={}):
    '''切片进程的参数
        1、首先获取默认参数，
        2、然后通过传入的参数更新已有的参数
        3、最后通过`options_post_processing`方法处理参数
    '''
    _options = DEFAULT_GDAL2TILES_OPTIONS.copy()
    _options.update(options)
    options = AttrDict(_options)
    options = options_post_processing(options, input_file, output_folder)
    return options

def generate_leaflet_overview(options, tminz, tmaxz, tilesize, tileext, swne):
    """
    Template for leaflet.html implementing overlay of tiles for 'mercator' profile.
    It returns filled string. Expected variables:
    title, north, south, east, west, minzoom, maxzoom, tilesize, tileformat, publishurl
    """

    args = {}
    args['title'] = options.title.replace('"', '\\"')
    args['htmltitle'] = options.title
    args['south'], args['west'], args['north'], args['east'] = swne
    args['centerlon'] = (args['north'] + args['south']) / 2.
    args['centerlat'] = (args['west'] + args['east']) / 2.
    args['minzoom'] = tminz
    args['maxzoom'] = tmaxz
    args['beginzoom'] = tmaxz
    args['tilesize'] = tilesize  # not used
    args['tileformat'] = tileext
    args['publishurl'] = options.url  # not used
    args['copyright'] = options.copyright.replace('"', '\\"')

    s = """<!DOCTYPE html>
    <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no' />
        <title>%(htmltitle)s</title>

        <!-- Leaflet -->
        <link rel="stylesheet" href="http://cdn.leafletjs.com/leaflet-0.7.5/leaflet.css" />
        <script src="http://cdn.leafletjs.com/leaflet-0.7.5/leaflet.js"></script>

        <style>
            body { margin:0; padding:0; }
            body, table, tr, td, th, div, h1, h2, input { font-family: "Calibri", "Trebuchet MS", "Ubuntu", Serif; font-size: 11pt; }
            #map { position:absolute; top:0; bottom:0; width:100%%; } /* full size */
            .ctl {
                padding: 2px 10px 2px 10px;
                background: white;
                background: rgba(255,255,255,0.9);
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                text-align: right;
            }
            .title {
                font-size: 18pt;
                font-weight: bold;
            }
            .src {
                font-size: 10pt;
            }

        </style>

    </head>
    <body>

    <div id="map"></div>

    <script>
    /* **** Leaflet **** */

    // Base layers
    //  .. OpenStreetMap
    var osm = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'});

    //  .. CartoDB Positron
    var cartodb = L.tileLayer('http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'});

    //  .. OSM Toner
    var toner = L.tileLayer('http://{s}.tile.stamen.com/toner/{z}/{x}/{y}.png', {attribution: 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'});

    //  .. White background
    var white = L.tileLayer("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEAAQMAAABmvDolAAAAA1BMVEX///+nxBvIAAAAH0lEQVQYGe3BAQ0AAADCIPunfg43YAAAAAAAAAAA5wIhAAAB9aK9BAAAAABJRU5ErkJggg==");

    // Overlay layers (TMS)
    var lyr = L.tileLayer('./{z}/{x}/{y}.%(tileformat)s', {tms: true, opacity: 0.7, attribution: "%(copyright)s"});

    // Map
    var map = L.map('map', {
        center: [%(centerlon)s, %(centerlat)s],
        zoom: %(beginzoom)s,
        minZoom: %(minzoom)s,
        maxZoom: %(maxzoom)s,
        layers: [osm]
    });

    var basemaps = {"OpenStreetMap": osm, "CartoDB Positron": cartodb, "Stamen Toner": toner, "Without background": white}
    var overlaymaps = {"Layer": lyr}

    // Title
    var title = L.control();
    title.onAdd = function(map) {
        this._div = L.DomUtil.create('div', 'ctl title');
        this.update();
        return this._div;
    };
    title.update = function(props) {
        this._div.innerHTML = "%(title)s";
    };
    title.addTo(map);

    // Note
    var src = 'Generated by <a href="http://www.klokan.cz/projects/gdal2tiles/">GDAL2Tiles</a>, Copyright &copy; 2008 <a href="http://www.klokan.cz/">Klokan Petr Pridal</a>,  <a href="http://www.gdal.org/">GDAL</a> &amp; <a href="http://www.osgeo.org/">OSGeo</a> <a href="http://code.google.com/soc/">GSoC</a>';
    var title = L.control({position: 'bottomleft'});
    title.onAdd = function(map) {
        this._div = L.DomUtil.create('div', 'ctl src');
        this.update();
        return this._div;
    };
    title.update = function(props) {
        this._div.innerHTML = src;
    };
    title.addTo(map);


    // Add base layers
    L.control.layers(basemaps, overlaymaps, {collapsed: false}).addTo(map);

    // Fit to overlay bounds (SW and NE points with (lat, lon))
    map.fitBounds([[%(south)s, %(east)s], [%(north)s, %(west)s]]);

    </script>

    </body>
    </html>

    """ % args    # noqa

    return s


def options_post_processing(options, input_file, output_folder):
    if not options.title:
        options.title = os.path.basename(input_file)

    if options.url and not options.url.endswith('/'):
        options.url += '/'
    if options.url:
        out_path = output_folder
        if out_path.endswith("/"):
            out_path = out_path[:-1]
        options.url += os.path.basename(out_path) + '/'

    if isinstance(options.zoom, (list, tuple)) and len(options.zoom) < 2:
        raise ValueError('Invalid zoom value')

    # Supported options
    if options.resampling == 'average':
        try:
            if gdal.RegenerateOverview:
                pass
        except Exception:
            exit_with_error("'average' resampling algorithm is not available.",
                            "Please use -r 'near' argument or upgrade to newer version of GDAL.")

    elif options.resampling == 'antialias':
        try:
            if numpy:     # pylint:disable=W0125
                pass
        except Exception:
            exit_with_error("'antialias' resampling algorithm is not available.",
                            "Install PIL (Python Imaging Library) and numpy.")

    try:
        os.path.basename(input_file).encode('ascii')
    except UnicodeEncodeError:
        full_ascii = False
    else:
        full_ascii = True

    # LC_CTYPE check
    if not full_ascii and 'UTF-8' not in os.environ.get("LC_CTYPE", ""):
        if not options.quiet:
            print("\nWARNING: "
                  "You are running gdal2tiles.py with a LC_CTYPE environment variable that is "
                  "not UTF-8 compatible, and your input file contains non-ascii characters. "
                  "The generated sample googlemaps, openlayers or "
                  "leaflet files might contain some invalid characters as a result\n")

    # Output the results
    if options.verbose:
        print("Options:", options)
        print("Input:", input_file)
        print("Output:", output_folder)
        print("Cache: %s MB" % (gdal.GetCacheMax() / 1024 / 1024))
        print('')

    return options


def generate_kml(tx, ty, tz, tileext, tilesize, tileswne, options, children=None, **args):
    """
    Template for the KML. Returns filled string.
    """
    if not children:
        children = []

    args['tx'], args['ty'], args['tz'] = tx, ty, tz
    args['tileformat'] = tileext
    if 'tilesize' not in args:
        args['tilesize'] = tilesize

    if 'minlodpixels' not in args:
        args['minlodpixels'] = int(args['tilesize'] / 2)
    if 'maxlodpixels' not in args:
        args['maxlodpixels'] = int(args['tilesize'] * 8)
    if children == []:
        args['maxlodpixels'] = -1

    if tx is None:
        tilekml = False
        args['title'] = options.title
    else:
        tilekml = True
        args['title'] = "%d/%d/%d.kml" % (tz, tx, ty)
        args['south'], args['west'], args['north'], args['east'] = tileswne(
            tx, ty, tz)

    if tx == 0:
        args['drawOrder'] = 2 * tz + 1
    elif tx is not None:
        args['drawOrder'] = 2 * tz
    else:
        args['drawOrder'] = 0

    url = options.url
    if not url:
        if tilekml:
            url = "../../"
        else:
            url = ""

    s = """<?xml version="1.0" encoding="utf-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>%(title)s</name>
        <description></description>
        <Style>
        <ListStyle id="hideChildren">
            <listItemType>checkHideChildren</listItemType>
        </ListStyle>
        </Style>""" % args
    if tilekml:
        s += """
    <Region>
      <LatLonAltBox>
        <north>%(north).14f</north>
        <south>%(south).14f</south>
        <east>%(east).14f</east>
        <west>%(west).14f</west>
      </LatLonAltBox>
      <Lod>
        <minLodPixels>%(minlodpixels)d</minLodPixels>
        <maxLodPixels>%(maxlodpixels)d</maxLodPixels>
      </Lod>
    </Region>
    <GroundOverlay>
      <drawOrder>%(drawOrder)d</drawOrder>
      <Icon>
        <href>%(ty)d.%(tileformat)s</href>
      </Icon>
      <LatLonBox>
        <north>%(north).14f</north>
        <south>%(south).14f</south>
        <east>%(east).14f</east>
        <west>%(west).14f</west>
      </LatLonBox>
    </GroundOverlay>
    """ % args

    for cx, cy, cz in children:
        csouth, cwest, cnorth, ceast = tileswne(cx, cy, cz)
        s += """
    <NetworkLink>
      <name>%d/%d/%d.%s</name>
      <Region>
        <LatLonAltBox>
          <north>%.14f</north>
          <south>%.14f</south>
          <east>%.14f</east>
          <west>%.14f</west>
        </LatLonAltBox>
        <Lod>
          <minLodPixels>%d</minLodPixels>
          <maxLodPixels>-1</maxLodPixels>
        </Lod>
      </Region>
      <Link>
        <href>%s%d/%d/%d.kml</href>
        <viewRefreshMode>onRegion</viewRefreshMode>
        <viewFormat/>
      </Link>
    </NetworkLink>
        """ % (cz, cx, cy, args['tileformat'], cnorth, csouth, ceast, cwest,
               args['minlodpixels'], url, cz, cx, cy)

    s += """      </Document>
    </kml>
    """
    return s