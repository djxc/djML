from dgdal2tiles.gdal2tiles import generate_tiles

if __name__ == "__main__":
    generate_tiles(r"E:\china_cloudless_4326.tif", r"E:\其他\tiles", resampling="near")
