from __future__ import division

import scipy
import json
from scipy.optimize import bracket, brentq, minimize
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import matplotlib as mpl
from precipitation import PrecipitationSimulator, PrecipitationLoader

import geopandas as gpd
import numpy as np
from scipy.ndimage import zoom

import math

import os

class FloodSimulator:
    def __init__(self, engine, generator):
        self.engine = engine
        self.generator = generator

        self.surfaceWater = np.zeros([engine.regionContext.height, engine.regionContext.width])
        self.lastSurfaceWater = np.zeros([engine.regionContext.height, engine.regionContext.width])
        self.maxSurfaceWater = np.zeros([engine.regionContext.height, engine.regionContext.width])

        self.writeImage = True

        pass

    def updateSurfaceWater(self, pattern):
        self.lastSurfaceWater = self.surfaceWater.copy()
        # print "lastSurfaceWater"
        # print np.sum(self.lastSurfaceWater)

        self.surfaceWater = pattern.copy()
        # print "surfaceWater"
        # print np.sum(self.surfaceWater)

    def createHazardFootprint(self):

        for s_i in range(0, self.engine.currentSimulationContext.numberTimeSteps):
            surfaceWater_dir = self.engine.dataContext.getImpactDirectory("Flood/") + str(s_i) + ".tif"

            tempSurfaceWater = self.engine.utilities.load_raster(surfaceWater_dir)
            self.maxSurfaceWater = np.maximum(self.maxSurfaceWater, tempSurfaceWater)


    def writeHazardFootprint(self):

        hazardMap_dir = self.engine.dataContext.getImpactDirectory(
            "/hazard_footprint_") + self.engine.currentSimulationContext.currentRun
        self.engine.utilities.save_raster(hazardMap_dir + ".tif", self.maxSurfaceWater)

        shapefile_path = self.engine.dataContext.sourceDataDirectory + "/Buildings/chur_buildings.shp"
        self.geodataToImage(shapefile_path, hazardMap_dir + ".tif", hazardMap_dir + '.png',
                            'Hazard Footprint: Flood ' + self.engine.currentSimulationContext.currentRun)

        outputLocation = self.engine.dataContext.getImpactDirectory("Flood/") + str(
            self.engine.currentSimulationContext.currentTimeStep)

        self.engine.utilities.save_raster(outputLocation + ".tif", self.surfaceWater)

        outputDict = {}

        outputDict["flood_sum"] = np.sum(self.maxSurfaceWater)
        outputDict["flood_max"] = np.amax(self.maxSurfaceWater)
        outputDict["flood_min"] = np.amin(self.maxSurfaceWater)
        outputDict["flood_avg"] = np.average(self.maxSurfaceWater)
        outputDict["flood_median"] = np.median(self.maxSurfaceWater)

        outfile = open(hazardMap_dir + ".json", 'w')
        json.dump(outputDict, outfile)


    def preprocess(self):
        self.generator.preprocess()

        pass

    def initialize(self):
        self.generator.initialize()
        pass

    def run(self):
        print(self.engine.utilities.getCurrentTime() + "-- flood: computing water depths --")
        self.updateSurfaceWater(self.generator.generateSurfaceWater())

        pass

    def writeResults(self, verbose = 0):
        outputLocation = self.engine.dataContext.getImpactDirectory("Flood/") + str(
            self.engine.currentSimulationContext.currentTimeStep)

        self.engine.utilities.save_raster(outputLocation + ".tif", self.surfaceWater)

        if verbose:
            outputDict = {}

            outputDict["flood_sum"] = np.sum(self.surfaceWater)
            outputDict["flood_max"] = np.amax(self.surfaceWater)
            outputDict["flood_min"] = np.amin(self.surfaceWater)
            outputDict["flood_avg"] = np.average(self.surfaceWater)
            outputDict["flood_median"] = np.median(self.surfaceWater)


            outfile = open(outputLocation + ".json", 'w')
            json.dump(outputDict, outfile)


        if self.writeImage == True:
            shapefile_path = self.engine.dataContext.sourceDataDirectory + "/Roads/roads.shp"
            flood_path = self.engine.dataContext.getImpactDirectory("Flood/") + str(self.engine.currentSimulationContext.currentTimeStep) + '.tif'
            image_dir = self.engine.dataContext.getImpactDirectory('Flood/') + 'Images/'

            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            outputLocation = image_dir + str(
                self.engine.currentSimulationContext.currentTimeStep) + '.png'

            image_title = "Inudation Map at Time Step " + str(
                self.engine.currentSimulationContext.currentTimeStep)

            self.geodataToImage(shapefile_path, flood_path,outputLocation,image_title)

            pass


        pass

    def geodataToImage(self, shapefile_path, flood_path, outputLocation, image_title, attribute='OBJEKTART',  ):
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
        try:
            shapefile_boundary = gpd.read_file(shapefile_path)
            is_shapefile = True
        except:
            is_shapefile = False
        # print(shapefile_boundary.crs)

        west_extent = self.engine.regionContext.west
        east_extent = self.engine.regionContext.west + self.engine.regionContext.width * self.engine.regionContext.resolution
        north_extent = self.engine.regionContext.north
        south_extent = self.engine.regionContext.north - self.engine.regionContext.height * self.engine.regionContext.resolution
        plot_extent = (west_extent, east_extent, south_extent, north_extent)

        dtm_data = self.generator.topography

        flood_data = self.engine.utilities.load_raster(flood_path)

        # Plot uncropped array
        figure, ax = plt.subplots()

        # ax.imshow(dtm_data, cmap='Greys', extent=plot_extent)

        if is_shapefile:
            filteredShapefile_data = shapefile_boundary
            filteredShapefile_data.plot(ax=ax,
                                    color='orange',
                                    # column=attribute,  categorical=True,
                                    # linewidth=np.sqrt(normalized_data) * 2,
                                    # vmin=0.0, vmax=4000.0,
                                    # legend=True,
                                    # legend_kwds={'label': attribute, 'orientation': 'horizontal', 'anchor': (0.25, 1.0),
                                    #              'pad': 0.05,
                                    #              'shrink': 0.6, 'aspect': 30},
                                    )

        flood_data[flood_data < 0] = 0

        # ax.imshow(filteredFlood_data, cmap='Blues', alpha = 0.9, extent = plot_extent)
        plt.imshow(flood_data, cmap='Blues', extent=plot_extent, vmin=30, vmax=50, alpha=0.9)
        plt.colorbar(ax=ax, label='Inundation Depth (mm)', orientation='vertical', location='right', pad=0.03, shrink=1,
                     aspect=30)


        ax.set_title(image_title)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # ax.axis('off')

        # plt.show()

        plt.savefig(outputLocation, dpi=300, bbox_inches='tight')
        plt.close()

    def postprocess(self):
        path = self.engine.dataContext.getConsequenceDirectory("Flood/")

        maxTimeStep = self.engine.damageSimulationContext.numberTimeSteps

        statsDict = {"flood_sum": [], "flood_max": [], "flood_min": [], "flood_avg": [], "flood_median": []}
        for ts in range(0, maxTimeStep):
            tspath = self.engine.dataContext.getConsequenceDirectory("Flood/") + str(ts) + ".json"

            with open(tspath) as data_file:
                data = json.load(data_file)
                statsDict["flood_sum"].append(data["flood_sum"])
                statsDict["flood_max"].append(data["flood_max"])
                statsDict["flood_min"].append(data["flood_min"])
                statsDict["flood_avg"].append(data["flood_avg"])
                statsDict["flood_median"].append(data["flood_median"])

        with open(path + "AggregateIndicators.json", 'w') as outJson:
            json.dump(statsDict, outJson, indent=4)

        pass

class TwoDFloodModel:
    def __init__(self,engine):
        self.engine = engine
        self.max_flow = 10
        self.return_period = self.engine.damageSimulationContext.return_period

    def initialize(self):

        # waterdepths should reflect the height of water at river
        self.surfaceWater = np.zeros([self.engine.regionContext.height, self.engine.regionContext.width])

        # flood discharge
        self.Q_inflow = 0  # in cubic meter per second


        # Inflow and outflow based on landcover
        landcover_shp_dir = self.engine.dataContext.sourceDataDirectory + '/Region/Landcover.shp'
        landcover_raster_dir = self.engine.dataContext.sourceDataDirectory + '/Region/Landcover.tif'

        dtm_dir =  self.engine.dataContext.sourceDataDirectory + '/dtm/dtm.tif'
        self.topography = self.engine.utilities.load_raster(dtm_dir)

        if not os.path.exists(landcover_raster_dir):
            self.engine.utilities.rasterizeShapefile(landcover_shp_dir, landcover_raster_dir, dtm_dir, setPixelValueFromAttributes = True,
                               attribute_name = 'type', all_touched = False)
        self.landcover = self.engine.utilities.load_raster(landcover_raster_dir)

        self.inflow_value_river_meterperhour = self.Q_inflow / self.engine.regionContext.resolution**2 * 3600 / 100

        self.outflow_value_river_meterperhour = self.inflow_value_river_meterperhour

        # number of iterations to move water between two adjacent cells in one hour; it is assumed to be 60 times (everyminute)
        self.numIteration = 60
        self.adjustedTimeInterval = 1 / self.numIteration

        # inflow from upstream cell
        self.I = np.zeros([self.engine.regionContext.height, self.engine.regionContext.width])

        pass

    def generateSurfaceWater(self):

        # add the precipitation
        upscale_precipitation = self.engine.precipitationSimulator.precipitationPattern

        # Assume raster_data is your original numpy array
        y_factor = self.topography.shape[0] / upscale_precipitation.shape[0]
        x_factor = self.topography.shape[1] / upscale_precipitation.shape[1]

        self.P = zoom(upscale_precipitation, (y_factor, x_factor))

        self.surfaceWater = self.surfaceWater + self.P

        for iter in range(self.numIteration):

            # Adding inflow to river
            inflow = (self.landcover == 1) * self.inflow_value_river_meterperhour * self.adjustedTimeInterval

            self.surfaceWater = self.surfaceWater + inflow
            distribution = self.gradientMethod()

            left_flow = distribution[0, :, :]
            right_flow = distribution[1, :, :]
            top_flow = distribution[2, :, :]
            bottom_flow = distribution[3, :, :]

            drain = np.sum(distribution, axis=0)

            # self.Qs = drain

            left_flow_pad = np.pad(left_flow[:, 1:], [(0, 0), (0, 1)], mode="constant", constant_values=0)
            right_flow_pad = np.pad(right_flow[:, :-1], [(0, 0), (1, 0)], mode="constant", constant_values=0)
            top_flow_pad = np.pad(top_flow[1:, :], [(0, 1), (0, 0)], mode="constant", constant_values=0)
            bottom_flow_pad = np.pad(bottom_flow[:-1, :], [(1, 0), (0, 0)], mode="constant", constant_values=0)

            # new inundation field
            self.I = left_flow_pad + right_flow_pad + top_flow_pad + bottom_flow_pad
            self.surfaceWater = self.surfaceWater + self.I - drain

            # Removing outflow to outlet
            outflow = (self.landcover == 2) * self.outflow_value_river_meterperhour * self.adjustedTimeInterval

            self.surfaceWater = self.surfaceWater - outflow
            self.surfaceWater[self.surfaceWater < 0] = 0

        self.surfaceWater[self.surfaceWater < 0] = 0
        return self.surfaceWater

    def gradientMethod(self):
        # max_water_distribution is defined as the maximum halved gradient
        # there should never be more water be redistributed than max_water_distribution
        # this ensures that the final inundation field converges to static values and thus avoids oscillation

        waterelevation = self.topography + self.surfaceWater

        gradient_left = (waterelevation[:, :-1] - waterelevation[:, 1:])
        gradient_right = (waterelevation[:, 1:] - waterelevation[:, :-1])
        gradient_top = (waterelevation[:-1, :] - waterelevation[1:, :])
        gradient_bottom = (waterelevation[1:, :] - waterelevation[:-1, :])

        gradient_left_pad = np.pad(gradient_left, [(0, 0), (1, 0)], mode="constant", constant_values=0)
        gradient_right_pad = np.pad(gradient_right, [(0, 0), (0, 1)], mode="constant", constant_values=0)
        gradient_top_pad = np.pad(gradient_top, [(1, 0), (0, 0)], mode="constant", constant_values=0)
        gradient_bottom_pad = np.pad(gradient_bottom, [(0, 1), (0, 0)], mode="constant", constant_values=0)

        gradients = np.stack((gradient_left_pad, gradient_right_pad, gradient_top_pad, gradient_bottom_pad))
        gradients[gradients > 0] = 0
        total_gradients = np.sum(gradients, axis=0)
        total_gradients[total_gradients == 0] = 1E14

        relative_gradients = (gradients / total_gradients) + 0.0

        max_flow_gradients = gradients * 0.5
        max_water_distribution = np.amax(np.abs(max_flow_gradients), axis=0)
        water_to_distribute = np.minimum(self.surfaceWater,max_water_distribution)
        distribution = (relative_gradients * water_to_distribute) + 0.0

        # free memory
        del waterelevation
        del gradient_left
        del gradient_right
        del gradient_top
        del gradient_bottom

        del gradient_left_pad
        del gradient_right_pad
        del gradient_top_pad
        del gradient_bottom_pad

        del gradients
        del max_flow_gradients
        del max_water_distribution
        del water_to_distribute
        del relative_gradients
        del total_gradients

        return distribution
