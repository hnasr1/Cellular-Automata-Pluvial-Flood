from osgeo import gdal, osr
import numpy as np

gdal.UseExceptions()

rasin = "C:/Users/hnasrazadani/polybox/Shared/CAT Project/Data/SourceData" + "/dtm/" + "dtm.tif"
shpin= "C:/Users/hnasrazadani/polybox/Shared/CAT Project/Data/SourceData" + "/Region/" + "regionMask.shp"
rasout = "C:/Users/hnasrazadani/polybox/Shared/CAT Project/Data/SourceData" + "/dtm/" + "dtm_clipped.tif"

# result = gdal.Warp(rasout, rasin, cutlineDSName = shpin)


dataset = gdal.Open(rasout)
input_grid = np.array(dataset.GetRasterBand(1).ReadAsArray(), dtype=float)
dataset = None

output_grid = input_grid[300:636, 804:1270]

height = len(output_grid)
width = len(output_grid[0])

dtm_final_path = "C:/Users/hnasrazadani/polybox/Shared/CAT Project/Data/SourceData" + "/dtm/" + "dtm_final.tif"
driver = gdal.GetDriverByName('GTiff')
ds = driver.Create(dtm_final_path, width, height, 1,
                   gdal.GDT_Float32)

band = ds.GetRasterBand(1)
band.WriteArray(output_grid, 0, 0)
band.FlushCache()
band.SetNoDataValue(-99)
band.GetStatistics(0, 1)

west = 741200 + 804 * 16
north = 199596 - 300 * 16

simpleGeoTransform = (west, 16, 0.0, north,0.0,-1 * 16)
ds.SetGeoTransform(simpleGeoTransform)
a = ds.GetGeoTransform()

target_crs = osr.SpatialReference()
target_crs.ImportFromEPSG(21781)

ds.SetProjection(target_crs.ExportToWkt())
a = ds.GetGeoTransform()

ds = None