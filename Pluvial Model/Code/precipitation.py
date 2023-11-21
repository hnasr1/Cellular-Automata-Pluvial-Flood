from __future__ import division
import numpy as np
import math as m
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
from scipy import ndimage, interpolate

from netCDF4 import Dataset
from osgeo import gdal
from osgeo import osr
import os
import json

import random
import math
# from networkx.algorithms.centrality.degree_alg import out_degree_centrality


class PrecipitationSimulator:
    def __init__(self, engine, generator):
        self.engine = engine
        self.generator = generator

        self.precipitationPattern = np.zeros([engine.regionContext.width, engine.regionContext.height])

        self.currentVertex = None
        pass

    def updatePrecipitation(self, pattern):
        self.precipitationPattern = pattern.copy()

    def run(self):
        print(self.engine.utilities.getCurrentTime() + "-- precipitation: generating precipitation pattern --")
        self.updatePrecipitation(self.generator.generatePrecipitationPattern())


        pass

    def initialize(self):
        self.generator.initialize()

        self.catchment_mask = (~np.isnan(
            self.engine.utilities.load_raster(self.engine.dataContext.sourceDataDirectory + "/dtm/dtm.tif"))).copy()

        pass

    def preprocess(self):
        self.generator.preprocess()
        pass

    def writeResults(self):
        outputLocation = self.engine.dataContext.getPrecipitationDirectory("Precipitation/") + str(
            self.engine.currentSimulationContext.currentTimeStep)
        self.engine.utilities.save_raster(outputLocation + ".tif", self.precipitationPattern)

        outputDict = {}
        outputDict["prec_sum"] = np.sum((self.precipitationPattern[self.catchment_mask] / 1000) * self.engine.regionContext.resolution * self.engine.regionContext.resolution)
        outputDict["prec_max"] = np.amax((self.precipitationPattern[self.catchment_mask] / 1000) * self.engine.regionContext.resolution * self.engine.regionContext.resolution)
        outputDict["prec_min"] = np.amin((self.precipitationPattern[self.catchment_mask] / 1000) * self.engine.regionContext.resolution * self.engine.regionContext.resolution)
        outputDict["prec_avg"] = np.average((self.precipitationPattern[self.catchment_mask] / 1000) * self.engine.regionContext.resolution * self.engine.regionContext.resolution)
        outputDict["prec_median"] = np.median((self.precipitationPattern[self.catchment_mask] / 1000) * self.engine.regionContext.resolution * self.engine.regionContext.resolution)

        outfile = open(outputLocation + ".json", 'w')
        json.dump(outputDict, outfile)

        # Commented till Further use - HN

        # self.vertex = self.engine.sg.addVertex(model_name="precipitation",
        #                                        time_step=self.engine.currentSimulationContext.currentTimeStep,
        #                                        state={"location": outputLocation + ".tif", "params": outputDict})

        pass

    def postprocess(self):
        directory = self.engine.dataContext.getPrecipitationDirectory("Precipitation/")

        maxTimeStep = self.engine.rainfallSimulationContext.numberTimeSteps

        statsDict = {"prec_sum": [], "prec_max": [], "prec_min": [], "prec_avg": [], "prec_median": []}
        for ts in range(0, maxTimeStep):
            path = directory + str(ts) + ".json"
                # directory + "/Precipitation/" + str(ts) + ".json"

            with open(path) as data_file:
                data = json.load(data_file)
                statsDict["prec_sum"].append(data["prec_sum"])
                statsDict["prec_max"].append(data["prec_max"])
                statsDict["prec_min"].append(data["prec_min"])
                statsDict["prec_avg"].append(data["prec_avg"])
                statsDict["prec_median"].append(data["prec_median"])

        for ts in range(maxTimeStep, self.engine.damageSimulationContext.numberTimeSteps):
            statsDict["prec_sum"].append(0)
            statsDict["prec_max"].append(0)
            statsDict["prec_min"].append(0)
            statsDict["prec_avg"].append(0)
            statsDict["prec_median"].append(0)

        with open(directory + "AggregateIndicators.json", 'w') as outJson:
            json.dump(statsDict, outJson, indent=4)
        pass

# this modules seems not be used
class PrecipitationLoader:
    def __init__(self, engine):
        self.engine = engine

        pass

    def initialize(self):
        pass

    def preprocess(self):

        pass

    def generatePrecipitationPattern(self):
        precipitationFile = self.engine.dataContext.simulationDataDirectory + ('/Precipitation/') + str(
            self.engine.currentSimulationContext.currentTimeStep) + ".tif"
        if os.path.isfile(precipitationFile):
            self.cloud = self.engine.utilities.load_raster(precipitationFile)
        else:
            self.cloud = np.zeros([self.engine.regionContext.height, self.engine.regionContext.width])

        return self.cloud
        pass
