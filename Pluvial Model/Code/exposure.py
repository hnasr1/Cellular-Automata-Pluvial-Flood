import math
from scipy.stats import lognorm

import numpy as np

import json
import csv
import ast
import copy
import os

from fiona import collection, open as fiona_open
from fiona.crs import from_epsg
from shapely.geometry import mapping, shape, Point, LineString
from shapely import wkt

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Exposure:
    def __init__(self, engine, ):
        self.engine = engine

        self.buildings = {}

        self.buildingsRequiringUpdate = set()

        pass

    def preprocess(self):
        buildings = self.engine.dataContext.sourceDataDirectory + "/Buildings/chur_buildings.shp"
        dtm = self.engine.dataContext.sourceDataDirectory + "/dtm/" + "dtm_final.tif"

        buildingsCSV = self.engine.dataContext.preprocessedDataDirectory + "/buildings.csv"
        cell2featCSV = self.engine.dataContext.preprocessedDataDirectory + "/cell2feat.csv"
        feat2edgeCSV = self.engine.dataContext.preprocessedDataDirectory + "/feat2bldg.csv"

        ria = self.engine.utilities.RasterIndexAssigner(buildings, dtm)

        cellIndicesToFeatureIDsMap = {}
        featureIDsToEdgeIDsMap = {}

        with open(buildingsCSV, 'w', newline='') as f:
            w = csv.DictWriter(f,
                               ["feat_id", "bldg_id", "type", "area", 'indices', 'geometry'])
            w.writeheader()
            with collection(buildings, "r") as features:
                for feature in features:
                    raster_Index_list = []

                    origType = feature['properties']["OBJEKTART"]

                    geometry = shape(feature['geometry'])
                    feat_id = feature["id"]
                    bldg_id = feature['properties']["UUID"]

                    # centroid point
                    i_point = geometry.centroid

                    indices = ria.fetchRasterDataByPoint(i_point.coords[0][0], i_point.coords[0][1], True)

                    # check if geometry is outside of extent, should not occur if properly prepared
                    if (indices[0] > self.engine.regionContext.width - 1 or indices[
                        1] > self.engine.regionContext.height - 1):
                        indices[0] = -1
                        indices[1] = -1

                    raster_Index_list.append(indices)


                    area = geometry.area

                    w.writerow(
                        {"feat_id": feat_id, "bldg_id": bldg_id, "type": origType, "area": area,
                         'indices': sorted(list(set(raster_Index_list))), 'geometry': geometry})

                    if indices in cellIndicesToFeatureIDsMap:
                        cellIndicesToFeatureIDsMap[indices].append(feat_id)
                    else:
                        cellIndicesToFeatureIDsMap[indices] = [feat_id]

                    featureIDsToEdgeIDsMap[feat_id] = bldg_id

        # write Index files
        with open(cell2featCSV, 'w', newline='') as f:
            w = csv.DictWriter(f, ["indices", "feat_ids"])
            w.writeheader()
            for key, value in cellIndicesToFeatureIDsMap.items():
                w.writerow({"indices": key, "feat_ids": value})

        with open(feat2edgeCSV, 'w', newline='') as f:
            w = csv.DictWriter(f, ["feat_id", "bldg_id"])
            w.writeheader()
            for key, value in featureIDsToEdgeIDsMap.items():
                w.writerow({"feat_id": key, "bldg_id": value})

        pass

    def initialize(self):
        buildingsCSV = self.engine.dataContext.preprocessedDataDirectory + "/buildings.csv"
        cell2featCSV = self.engine.dataContext.preprocessedDataDirectory + "/cell2feat.csv"
        feat2bldgCSV = self.engine.dataContext.preprocessedDataDirectory + "/feat2bldg.csv"


        self.buildingsUnitCost = json.load(
            open(self.engine.dataContext.sourceDataDirectory + "/Buildings/buildingsUnitCost.json"))

        print(self.engine.utilities.getCurrentTime() + "-- buildings: loading cell2feat.csv --")
        self.cell2featMap = {}
        with open(cell2featCSV, 'r', newline='') as cell2featCSVFile:
            IndexReader = csv.DictReader(cell2featCSVFile, delimiter=',')
            for row in IndexReader:
                self.cell2featMap[ast.literal_eval(row["indices"])] = ast.literal_eval(row["feat_ids"])

        print(self.engine.utilities.getCurrentTime() + "-- exposure: loading feat2bldg.csv --")
        self.feat2bldgMap = {}
        with open(feat2bldgCSV, 'r', newline='') as feat2bldgCSVFile:
            IndexReader = csv.DictReader(feat2bldgCSVFile, delimiter=',')
            for row in IndexReader:
                self.feat2bldgMap[ast.literal_eval(row["feat_id"])] = str(row["bldg_id"])

        print(self.engine.utilities.getCurrentTime() + "-- infrastructure: loading buildings.csv --")

        with open(buildingsCSV, 'r', newline='') as buildingsCSVFile:
            featuresReader = csv.DictReader(buildingsCSVFile, delimiter=',')
            for row in featuresReader:
                bldg_id = str(row["bldg_id"])
                feat_id = int(row["feat_id"])
                area = float(row["area"])
                bldg_type = row["type"]
                geometry = row["geometry"]
                bldg_unit_cost = self.buildingsUnitCost[bldg_type]
                indices = ast.literal_eval(row["indices"])[0]

                building = Building(self.engine, feat_id, bldg_id, indices, area, geometry, bldg_type, bldg_unit_cost)

                building.impactEvaluator = ImpactEvaluator(self.engine,building,)

                self.buildings[feat_id] = building

                pass

        pass

    def setIntensityMeasure(self, feat_id, measure):

        building = self.buildings[int(feat_id)]

        building.impactEvaluator.updateState(measure)

        self.buildingsRequiringUpdate.add(feat_id)

        pass

    def updateBuildings(self):
        for feat_id in self.buildingsRequiringUpdate:
            self.buildings[int(feat_id)].updateState()

        pass

    def resetUpdateLists(self):
        self.buildingsRequiringUpdate.clear()

        pass

    def writeResults(self):

        format = "ESRI Shapefile"
        suffix = ".shp"

        baseName = self.engine.dataContext.getImpactDirectory("Impact/")

        print(self.engine.utilities.getCurrentTime() + "-- exposure: writing impact properties --")
        self.exportToGeoFile(baseName + "impactProperties", suffix, format, ImpactEvaluator.getStateDescription(), lambda building: building.impactEvaluator.getState())
        self.exportToCSVFile(baseName + "impactProperties.csv", ImpactEvaluator.getStateDescription(),
                             lambda building: building.impactEvaluator.getState())

        print(self.engine.utilities.getCurrentTime() + "-- exposure: writing building aggregation properties --")
        self.exportBuildingIndicators(baseName + "buildingIndicators",
                                     {"buildingAggregators": Building.getAggregators(),
                                      "inundationHazardAggregators": ImpactEvaluator.getAggregators(),})


        impactLocation = self.engine.dataContext.getImpactDirectory("Impact/") +"impactProperties.shp"
        hazard_footprint_dir = self.engine.dataContext.getImpactDirectory("/hazard_map_") + \
                               self.engine.damageSimulationContext.currentRun + '.tif'

        self.engine.utilities.geodataToImage(impactLocation, 1, hazard_footprint_dir, attribute = 'damage_rat')
        self.engine.utilities.geodataToImage(impactLocation, 10000, hazard_footprint_dir, attribute = 'exp_rc')


    def exportBuildingIndicators(self, path, aggregators):

        # initialize aggregation variables
        aggregation_values = {}
        for aggregator_group_id, aggregator_group in aggregators.items():
            aggregation_values[aggregator_group_id] = {}
            for aggregator_id, aggregator in aggregator_group.items():
                aggregation_values[aggregator_group_id][aggregator_id] = 0

        # fill aggregation variables
        for feat_id, building in self.buildings.items():
            for aggregator_group_id, aggregator_group in aggregators.items():
                for aggregator_id, aggregator in aggregator_group.items():
                    value = aggregator(building, aggregation_values[aggregator_group_id][aggregator_id])
                    aggregation_values[aggregator_group_id][aggregator_id] = value
                    # print "aggregator_group_id: " + str(aggregator_group_id) + " aggregator_id: " + str(aggregator_id) + " value: " + str(value)

        with open(path + ".json", 'w') as s:
            json.dump(aggregation_values, s)

        pass

    def exportToGeoFile(self, location, suffix, format, properties, func):
        # append IDs to dictionary
        schema = {
            'geometry': 'Polygon',
            'properties': dict(list(properties.items()) + list({"feat_id": "int", "bldg_id": "str"}.items()))
        }

        nonValue = -999

        with fiona_open(location + suffix, 'w', format, schema, crs=from_epsg(self.engine.regionContext.epsg)) as c:
            for feat_id, building in self.buildings.items():
                ids = {"feat_id": feat_id, "bldg_id": building.id}

                state = func(building)
                if state == None:
                    state = {key: nonValue for key in properties.keys()}
                propertiesDict = dict(list(ids.items()) + list(state.items()))

                mappedGeometry = mapping(wkt.loads(building.geometry))

                # print "intersection " + str(set(schemaSections["properties"].keys()) ^ set(propertiesDict.keys()))

                c.write({"geometry": mappedGeometry,
                         "properties": propertiesDict})

        pass

    def exportToCSVFile(self, location, properties, func):
        # combDict = dict(properties.items() + {"feat_id" : "int", "edge_id" : "str", "geometry" : "wkt"}.items())
        combDict = dict(list(properties.items()) + list({"feat_id": "int", "bldg_id": "str"}.items()))
        nonValue = -999

        # dict.keys() and dict.values() will directly return in the same order (http://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order)
        with open(location, 'w', newline='') as fp:
            a = csv.writer(fp, delimiter=",")
            a.writerow([key for key in sorted(combDict)])

            for feat_id, building in self.buildings.items():
                ids = {"feat_id": feat_id, "bldg_id": building.id}

                state = func(building)
                if state == None:
                    state = {key: nonValue for key in properties.keys()}

                # propertiesDict = dict(ids.items() + state.items() + {"geometry" : mapping(wkt.loads(section.geometry))}.items())
                propertiesDict = dict(list(ids.items()) + list(state.items()))

                a.writerow([propertiesDict[key] for key in sorted(propertiesDict)])

        pass

class Building():
    @staticmethod
    def getStateDescription():
        return {"damage_ratio" : "float", "exp_rc" : "float" }
        pass

    @staticmethod
    def getAggregators():
        return {
                "sum exp_rc" : lambda building, aggregate: aggregate + max(0, building.exp_rc),
                }

    def __init__(self, engine, feat_id, bldg_id,  indices, area, geometry, bldg_type, bldg_unit_cost):
        self.engine = engine
        self.id = bldg_id
        self.fid = feat_id
        self.indices = indices
        self.area = area
        self.bldg_type = bldg_type
        self.bldg_unit_cost = bldg_unit_cost
        self.geometry = geometry

        self.initialize()

        pass


    def initialize(self):
        # provided by damage models

        self.damage_ratio = 0
        self.exp_rc = 0

        pass


    def updateState(self):

        self.damage_ratio = self.impactEvaluator.damage_ratio
        self.exp_rc = self.impactEvaluator.exp_rc

        pass


    def getState(self):

        return {"damage_ratio" : self.damage_ratio, "exp_rc" : self.exp_rc  }

        pass


class ImpactEvaluator():

    @staticmethod
    def getAggregators():
        return {
            "count inundated": lambda building,
                                      aggregate: aggregate + 1 if building.impactEvaluator.inundationDepth > 0 else aggregate,
            "area inundated": lambda building, aggregate: aggregate + building.area if building.impactEvaluator.inundationDepth > 0 else aggregate,
        }
        pass

    @staticmethod
    def getStateDescription():
        return {"inundation": "float", "damage_ratio": "float", "exp_rc": "float"}
        pass

    def __init__(self, engine, building):
        self.name = "InundationImpactEvaluator"
        self.engine = engine
        self.building = building
        self.initialize()

        self.critical_building = [""]
        self.non_critical_building = ["",""]

        pass

    def initialize(self):
        self.inundationDepth = 0

        # impact measures
        self.damage_ratio = 0
        self.exp_rc = 0

        pass

    def updateState(self, measure):

        self.inundationDepth = measure
        height = (1 + 1 / 4) * 2.5  # with the assumption of having one floor upper ground (1/4 height) and one basement (1 height)

        # New inundation damage model
        if self.building.bldg_type in self.critical_building:
            unit_temp_exp_rc = 900 * np.sqrt(self.inundationDepth / 0.25)
            temp_exp_rc = unit_temp_exp_rc * self.building.area * (height)
            pass

        else:  # if the building is NOT critical
            unit_temp_exp_rc = 400 * np.sqrt(self.inundationDepth/0.25)
            temp_exp_rc = unit_temp_exp_rc * self.building.area * (height)
            pass

        # measures are in CHF/km
        rebuilding_cost = (self.building.bldg_unit_cost * self.building.area)
        self.exp_rc = min(temp_exp_rc, rebuilding_cost)
        self.damage_ratio = self.exp_rc / rebuilding_cost

        pass

    def getState(self):
        return {"inundation": self.inundationDepth, "damage_ratio": self.damage_ratio, "exp_rc": self.exp_rc, }
        pass

    def restore(self):
        self.initialize()
        pass

class ExposureDamageSimulator:
    def __init__(self, engine, exposure):
        self.engine = engine
        self.exposure = exposure

        pass

    def preprocess(self):
        pass

    def initialize(self):
        pass

    def getUpdatedInnundatedCellIndices(self):
        initial_surfaceWater = np.zeros([self.engine.regionContext.height, self.engine.regionContext.width])
        pathToSurfaceWater = self.engine.dataContext.getImpactDirectory("/hazard_map_") + self.engine.currentSimulationContext.currentRun + '.tif'
        surfaceWater = self.engine.utilities.load_raster(pathToSurfaceWater)
        surfaceWater[surfaceWater < 0] = 0

        self.engine.floodSimulator.surfaceWater = surfaceWater

        cellIndicesSurfaceWater = np.argwhere(surfaceWater != initial_surfaceWater)
        return cellIndicesSurfaceWater

    def run(self):

        cellIndicesSurfaceWater = self.getUpdatedInnundatedCellIndices()

        print(self.engine.utilities.getCurrentTime() + "-- exposure: updating damage states for buildings due to flooding --")
        for indices in cellIndicesSurfaceWater:
            if (indices[1], indices[0]) not in self.exposure.cell2featMap:
                continue

            im = self.engine.floodSimulator.surfaceWater[indices[0], indices[1]]

            #find the id of the features that are in cells where there is inundation
            if im > 0:
                ids = self.exposure.cell2featMap[(indices[1], indices[0])]
                for id in ids:
                    self.exposure.setIntensityMeasure(id, im)

        print(self.engine.utilities.getCurrentTime() + "-- exposure: updating states of buildings --")
        self.exposure.updateBuildings()
        self.exposure.resetUpdateLists()

        return
        pass
