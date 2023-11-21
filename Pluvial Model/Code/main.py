import sys

import numpy as np
import pandas as pd
import csv
import pickle
import math
import os
import json
import uuid
import shutil

from scipy.stats import norm
from time import strftime

from osgeo.gdalconst import *
from osgeo import osr, ogr, gdal
import geopandas as gpd
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib as mpl

from fiona import collection
from shapely.geometry import shape, LineString

from flood import FloodSimulator, TwoDFloodModel
from precipitation import PrecipitationSimulator, PrecipitationLoader
from exposure import *

class Engine:
    def __init__(self):

        self.damageSimulationContext = SimulationContext(self)

        self.currentSimulationContext = self.damageSimulationContext

        configPath = os.path.dirname(os.path.realpath(__file__)) + "\CAT_Model.conf"

        configFile = open(configPath)
        data = json.load(configFile)

        regionConfig = data["RegionContext"]
        dataConfig = data["DataContext"]
        self.regionContext = RegionContext(self, width=regionConfig["width"], height=regionConfig["height"],
                                             west=regionConfig["west"], north=regionConfig["north"],
                                             resolution=regionConfig["resolution"], epsg=regionConfig["epsg"])
        self.dataContext = DataContext(self, sourceDataDirectory=str(dataConfig["sourceDataDirectory"]),
                                         preprocessedDataDirectory=str(dataConfig["preprocessedDataDirectory"]),
                                         simulationDataDirectory=str(dataConfig["simulationDataDirectory"]))

        self.utilities = Utilities(self)


        pass

    def preprocess(self):
        self.exposure.preprocess()

    def simulateHazard(self):
        print(self.utilities.getCurrentTime() + "------------ engine: simulating hazard  ------------")
        
        for s_i in range(0, self.currentSimulationContext.numberTimeSteps):
            print(self.utilities.getCurrentTime() + "------------ simulation time step: " + str(self.currentSimulationContext.currentTimeStep) + " ------------")
            self.precipitationSimulator.run()
            self.floodSimulator.run()
            self.floodSimulator.writeResults()

            self.currentSimulationContext.currentTimeStep += 1

    def createHazardFootprint(self):
        print(self.utilities.getCurrentTime() + "------------ engine: creating hazard map ------------")
        self.floodSimulator.createHazardFootprint()
        self.floodSimulator.writeHazardFootprint()

    def computeImpact(self):
        print(self.utilities.getCurrentTime() + "------------ engine: computing impact ------------")
        self.exposureDamageSimulator.run()
        self.exposure.writeResults()

    def setUpDirectories(self, baseDir, dirs=["Flood", "Exposure"]):
        print(self.utilities.getCurrentTime() + "-- engine: creating output directories at " + str(baseDir) + " --")


        for dir in dirs:
            pathToCreate = baseDir + '/' + dir + '/'

            # this is a coarse and incomplete check that prevents to accidently clean root, i.e. "/", or any other base directories (hopefully).
            if not any(c.isalpha() for c in pathToCreate) and not (pathToCreate == "/"):
                print ("WARNING: DIRECTORY TO BE CREATED DOES NOT CONTAIN ANY ALPHABETIC LETTER. FOR SECURITY REASONS THE DIRECTORY WILL NEITHER BE CLEANED NOR CREATED.")
                continue

            if not os.path.exists(pathToCreate):
                os.makedirs(pathToCreate)
                print("creating: " + str(pathToCreate))

        pass

def main(argv):
    if len(argv) == 0:
        print('-- no arguments given: quitting --')

        return

    # load config file
    engine = Engine()

    if argv[0] == "preprocess":
        engine.exposure = Exposure(engine)
        engine.preprocess()

    if argv[0] == "hazard":

        engine.damageSimulationContext.currentRun = "test"
        engine.damageSimulationContext.currentTimeStep = 0
        engine.damageSimulationContext.numberTimeSteps = 1000

        engine.setUpDirectories(engine.dataContext.getImpactDirectory(),["Flood"])

        engine.precipitationSimulator = PrecipitationSimulator(engine, PrecipitationLoader(engine))
        engine.floodSimulator = FloodSimulator(engine, TwoDFloodModel(engine))


        # initialize simulation modules
        engine.precipitationSimulator.initialize()
        engine.floodSimulator.initialize()

        # start simulation
        engine.currentSimulationContext = engine.damageSimulationContext
        engine.simulateHazard()
        engine.createHazardFootprint()

    if argv[0] == "hazardCurve":
        engine.floodSimulator = FloodSimulator(engine, TwoDFloodModel(engine))
        engine.floodSimulator.initialize()

        sim_dir = engine.dataContext.simulationDataDirectory
        sub_folders = [name for name in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, name))]
        all_hazard_footprints = {}
        for folder in sub_folders:
            return_period = int(folder.split('_')[2])
            hazardFootprintDir = sim_dir + '/' + folder + '/hazard_map_' + folder + '.tif'
            hazardFottprint = engine.utilities.load_raster(hazardFootprintDir)

            all_hazard_footprints[return_period] = hazardFottprint

        depth_range = np.arange(0, 50.5, 0.5)
        probability_range_map = [np.zeros_like(hazardFottprint) for _ in range(len(depth_range))]
        weigted_depth_map = np.ones_like(probability_range_map)

        for i in range(len(depth_range)):
            d =  depth_range[i]

            for key in all_hazard_footprints:
                return_period = key
                prob = 1/return_period
                probability_range_map[i] += prob * (1-np.heaviside(d - all_hazard_footprints[key], 1))

            weigted_depth_map[i] = d * probability_range_map[i]

        probability_range_map_stacked = np.stack(probability_range_map)
        prob_list = probability_range_map_stacked.reshape(len(depth_range),-1).T.reshape(336,466,len(depth_range))
        np.save(sim_dir + "/hazardCurves.npy", prob_list)

        x1 = 321
        y1 = 15

        x2 = 298
        y2 = 60

        df_example_probs = pd.DataFrame(columns=['water depth', 'point 1', 'point 2'])
        probs1 = np.zeros_like(depth_range)
        probs2 = np.zeros_like(depth_range)

        for i in range(len(depth_range)):
            d1 =  probability_range_map[i][x1][y1]
            d2 =  probability_range_map[i][x2][y2]

            probs1[i] = d1
            probs2[i] = d2

        df_example_probs['water depth'] = depth_range
        df_example_probs['probability point 1'] = probs1
        df_example_probs['probability point 2'] = probs2

        sample_dir = sim_dir + '/sample_prob.csv'
        df_example_probs.to_csv(sample_dir)

    if argv[0] == "hazardMap":
        return_period = float(argv[1])
        engine.floodSimulator = FloodSimulator(engine, TwoDFloodModel(engine))
        engine.floodSimulator.initialize()
        scenario_prob = 1 / return_period
        sim_dir = engine.dataContext.simulationDataDirectory

        hazard_map = np.zeros([engine.regionContext.height, engine.regionContext.width])
        prob_list = np.load(sim_dir + "/hazardCurves.npy")
        depth_range = np.arange(0, 50.5, 0.5)

        for i in range(prob_list.shape[0]):
            for j in range(prob_list.shape[1]):
                probL = prob_list[i][j]
                for k in range(len(probL)):
                    p = probL[k]
                    if p < scenario_prob:
                        break

                hazard_map[i,j] = depth_range[k]


        hazard_map_dir = sim_dir + '/hazard_map_' + argv[1] + '.tif'
        hazard_map_image_dir = sim_dir + '/hazard_map_'+ argv[1]  + '.png'

        engine.utilities.save_raster(hazard_map_dir, hazard_map)

        shapefile_path = engine.dataContext.sourceDataDirectory + "/Buildings/chur_buildings.shp"
        engine.floodSimulator.geodataToImage(shapefile_path, hazard_map_dir, hazard_map_image_dir,
                                             "Hazard Map T = " + argv[1])

    if argv[0] == "impact":
        return_period = float(argv[1])

        engine.damageSimulationContext.currentRun = 'test'
        engine.damageSimulationContext.return_period = return_period

        engine.setUpDirectories(engine.dataContext.getImpactDirectory(),["Impact"])

        engine.floodSimulator = FloodSimulator(engine, TwoDFloodModel(engine))
        engine.exposure = Exposure(engine)
        engine.exposureDamageSimulator = ExposureDamageSimulator(engine, engine.exposure)


        # initialize simulation modules
        engine.floodSimulator.initialize()
        engine.exposure.initialize()
        engine.exposureDamageSimulator.initialize()

        # start simulation
        engine.currentSimulationContext = engine.damageSimulationContext
        engine.computeImpact()

    if argv[0] == 'aggregateIndicators':
        df = pd.DataFrame(columns=["return period", "sum exp_rc", "count inundated", "area inundated"])
        sim_dir = engine.dataContext.simulationDataDirectory
        sub_folders = [name for name in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, name))]
        print(sub_folders)

        for folder in sub_folders:
            indicatorFile = sim_dir + '/' + folder + '/Impact/buildingIndicators.json'
            if os.path.exists(indicatorFile):
                with open(indicatorFile) as json_file:
                    data = json.load(json_file)

                    return_period = int(folder.split('_')[2])
                    sum_exp_rc = data['buildingAggregators']['sum exp_rc']
                    count_inundated = data['inundationHazardAggregators']['count inundated']
                    area_inundated = data['inundationHazardAggregators']['area inundated']
                    df.loc[len(df.index)] = [return_period, sum_exp_rc, count_inundated, area_inundated]

                    # print(data)


        df.to_csv(sim_dir + 'ResultsSummary.csv')

    pass


class RegionContext:
    def __init__(self, engine, width=1600, height=1200, west=741200, north=199596, resolution=16, epsg=21781):
        self.engine = engine

        self.width = width
        self.height = height
        self.west = west
        self.north = north
        self.resolution = resolution
        self.epsg = epsg

    pass


class DataContext:
    def __init__(self, engine, sourceDataDirectory="",
                 preprocessedDataDirectory="",
                 simulationDataDirectory="",):

        self.engine = engine

        self.sourceDataDirectory = sourceDataDirectory
        self.preprocessedDataDirectory = preprocessedDataDirectory
        self.simulationDataDirectory = simulationDataDirectory
        pass

    def getSimulationRunDataDirectory(self):
        return self.simulationDataDirectory + "/" + str(self.engine.currentSimulationContext.currentRun) + "/"

    def getImpactDirectory(self, subfolder=''):
        return self.simulationDataDirectory + self.engine.damageSimulationContext.currentRun + "/" + subfolder

class SimulationContext:
    def __init__(self, engine, timeStep=3600, numberTimeSteps=1, currentRun=1, currentTimeStep=0, return_period = 500,
                 name="damage_simulation"):
        self.engine = engine

        self.timeStep = timeStep  # in seconds; 900 equals 15 minutes
        self.numberTimeSteps = numberTimeSteps  # in seconds
        self.currentRun = currentRun
        self.currentTimeStep = currentTimeStep
        self.return_period = return_period


        self.name = name

    pass

class Utilities:
    def __init__(self, engine):
        self.engine = engine

    def load_raster(self, path_to_file):
        # Load Raster Data START
        dataset = gdal.Open(path_to_file, GA_ReadOnly)
        input_grid = np.array(dataset.GetRasterBand(1).ReadAsArray(), dtype=float)
        dataset = None
        return (input_grid)

    def save_raster(self, path_to_file, output_grid):
        # get parameters
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(path_to_file, self.engine.regionContext.width, self.engine.regionContext.height, 1,
                           GDT_Float32)

        band = ds.GetRasterBand(1)
        band.WriteArray(output_grid, 0, 0)
        band.FlushCache()
        band.SetNoDataValue(-99)
        band.GetStatistics(0, 1)

        simpleGeoTransform = (
            self.engine.regionContext.west, self.engine.regionContext.resolution, 0.0, self.engine.regionContext.north,0.0,-1 * self.engine.regionContext.resolution)
        ds.SetGeoTransform(simpleGeoTransform)

        target_crs = osr.SpatialReference()
        target_crs.ImportFromEPSG(self.engine.regionContext.epsg)

        ds.SetProjection(target_crs.ExportToWkt())
        ds = None

        pass

    def getCurrentTime(self):
        return str(strftime("%Y-%m-%d %H:%M:%S"))
        pass

    def geodataToImage(self, shapefile_path, range_max, flood_path = '', attribute='exp_rc'):
            plt.clf()
            # plt.ion()

            # Set figure size and title size of plots
            mpl.rcParams['figure.figsize'] = (
            self.engine.regionContext.width / 100 + 2, self.engine.regionContext.height / 100)
            mpl.rcParams['axes.titlesize'] = 12
            mpl.rcParams['axes.labelsize'] = 10
            mpl.rcParams['xtick.labelsize'] = 9
            mpl.rcParams['ytick.labelsize'] = 9

            # Open fire boundary data with geopandas
            shapefile_boundary = gpd.read_file(shapefile_path)

            # print(shapefile_boundary.crs)

            west_extent = self.engine.regionContext.west
            east_extent = self.engine.regionContext.west + self.engine.regionContext.width * self.engine.regionContext.resolution
            north_extent = self.engine.regionContext.north
            south_extent = self.engine.regionContext.north - self.engine.regionContext.height * self.engine.regionContext.resolution
            plot_extent = (west_extent, east_extent, south_extent, north_extent)


            # Plot uncropped array
            figure, ax = plt.subplots()

            shapefile_boundary.plot(ax=ax, cmap='Oranges',
                                        column=attribute,  # categorical=True,

                                        vmin=0.0, vmax=range_max,
                                        legend=True,
                                        legend_kwds={'label': attribute, 'orientation': 'horizontal', 'anchor': (0.6, 1.0),
                                                     'pad': 0.05,'shrink': 0.6, 'aspect': 30},
                                        )

            flood_data = self.engine.utilities.load_raster(flood_path)

            # ax.imshow(filteredFlood_data, cmap='Blues', alpha = 0.9, extent = plot_extent)
            plt.imshow(flood_data, cmap='Blues', extent=plot_extent, vmin=0, vmax=5, alpha=0.9)
            plt.colorbar(ax=ax, label='Inundation Depth (m)', orientation='vertical', location='right', pad=0.03, shrink=1,
                         aspect=30)


            ax.set_title("Impact Map")
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            # ax.axis('off')

            image_dir = self.engine.dataContext.getImpactDirectory()

            outputLocation = image_dir + 'impact_' + attribute + '.png'

            # plt.show()

            plt.savefig(outputLocation, dpi=300, bbox_inches='tight')
            plt.close()

    def rasterizeShapefile(self, InputVector, OutputImage, RefImage, setPixelValueFromAttributes=True,
                           attribute_name='Height', all_touched=False):

        ALL_TOUCHED = 'FALSE'
        if all_touched:
            ALL_TOUCHED = "TRUE"

        gdalformat = 'GTiff'
        datatype = gdal.GDT_Float32
        burnVal = 1  # value for the output image pixels
        ##########################################################
        # Get projection info from reference image
        Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

        # Open Shapefile
        try:
            Shapefile = ogr.Open(InputVector)
            Shapefile_layer = Shapefile.GetLayer()
        except:
            Shapefile_layer = InputVector

        field_vals = []
        feature = Shapefile_layer.GetNextFeature()

        while feature:
            field_vals.append(feature.GetFieldAsString(attribute_name))
            feature = Shapefile_layer.GetNextFeature()

        # Rasterise
        print("Rasterising shapefile...")
        Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1,
                                                         datatype,
                                                         options=['COMPRESS=DEFLATE'])
        Output.SetProjection(Image.GetProjectionRef())
        Output.SetGeoTransform(Image.GetGeoTransform())

        # Write data to band 1
        Band = Output.GetRasterBand(1)
        Band.SetNoDataValue(0)

        if setPixelValueFromAttributes:
            gdal.RasterizeLayer(Output, [1], Shapefile_layer,
                                options=["ALL_TOUCHED=" + ALL_TOUCHED, "ATTRIBUTE=" + attribute_name])
        else:
            gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal],
                                options=["ALL_TOUCHED=" + ALL_TOUCHED])

        Output.FlushCache()

        input_grid = np.array(Output.GetRasterBand(1).ReadAsArray(), dtype=float)
        a = np.array(Band.ReadAsArray(), dtype=float)

        # Close datasets
        Band = None
        Output = None
        Image = None
        Shapefile = None

        # Build image overviews
        subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE " + OutputImage + " 2 4 8 16 32 64", shell=True)
        # print("Done.")

    class RasterIndexAssigner:
        def __init__(self, input_shapefile, input_raster):
            self.input_shapefile = input_shapefile
            self.input_raster = input_raster

            self.src_ds = gdal.Open(self.input_raster)
            self.gt = self.src_ds.GetGeoTransform()
            self.rb = self.src_ds.GetRasterBand(1)

            pass

        def fetchRasterDataByPoint(self, x, y, return_Index=False):

            px = int((x - self.gt[0]) / self.gt[1])  # x pixel
            py = int((self.gt[3] - y) / -self.gt[5])  # y pixel

            if return_Index == True:
                return (px, py)

            data = self.rb.ReadAsArray(px, py, 1, 1)
            return float(data[0, 0])

if __name__ == "__main__":
    main(sys.argv[1:])
