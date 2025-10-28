import os
import shutil
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import csv
import pickle

class Logger():
    basePath = "./log"
    def __init__(self) -> None:
        """Init method that creates log directory if it does not exist."""
        if not os.path.exists("./log"):
            os.mkdir("./log")
        
        if not os.path.exists("./pickle_objects"):
            os.mkdir("./pickle_objects")
    
    def createLogFile(self, map, width, height, algorithm, crossover, mutation, popsize, n_eval, samplingFunction, repairFunction, shiftingMethod, seed, totalTime):
        """Creates a logfile for the path."""
        self.logName = f"{map.name}_{width}_{height}_{algorithm.__class__.__name__}_{crossover.__class__.__name__}_{mutation.__class__.__name__}_{popsize}_{n_eval}_{samplingFunction.__class__.__name__}_{repairFunction.__class__.__name__}_{shiftingMethod}_{seed}"
        self.logPath = self.basePath + "/" + self.logName
        # If a log for this already exists delete it
        if os.path.exists(self.logPath):
            shutil.rmtree(self.logPath)
        os.mkdir(self.logPath)

        #Check if csv exists
        if not os.path.exists("./log/results.csv"):
            with open("./log/results.csv", "w") as f:
                #We add the header of the file here
                header = "map, width, height, algorithm, crossover, mutation, popsize, n_eval, samplingFunction, repairFunction, shiftingMethod, seed, numberOfNonDominated, steps, shiftedWeight, paths, time\n"
                f.write(header)
                f.close()


        # Set class variables
        self.map = map.name
        self.width = width
        self.height = height
        self.algorithm = algorithm.__class__.__name__
        self.crossover = crossover.__class__.__name__
        self.mutation = mutation.__class__.__name__
        self.popsize = popsize
        self.n_eval = n_eval
        self.samplingFunction = samplingFunction.__class__.__name__
        self.repairFunction = repairFunction.__class__.__name__
        self.seed = seed
        self.time = totalTime

        #TODO: Make this better, just temporary solution
        if shiftingMethod == 0:
            self.shiftingMethod = "random"
        elif shiftingMethod == 1:
            self.shiftingMethod = "leastRestiance"
        elif shiftingMethod == 2:
            self.shiftingMethod = "splitInHalfShift"
        elif shiftingMethod == 3:
            self.shiftingMethod = "splitInThirdsShift"
        elif shiftingMethod == 4:
            self.shiftingMethod = "maxResistanceShift"
        else:
            self.shiftingMethod = "equalizeNeighborWeights"

    def log(self, paths, steps, shiftedWeight):
        """Creates a log object and writes it to the json file."""
        log_obj = {
            "map": self.map,
            "width": self.width,
            "height": self.height,
            "algorithm": self.algorithm,
            "crossover": self.crossover,
            "mutation": self.mutation,
            "popsize": self.popsize,
            "n_eval": self.n_eval,
            "samplingFunction": self.samplingFunction,
            "repairFunction": self.repairFunction,
            "shiftingMethod": self.shiftingMethod,
            "seed": self.seed,
            "numberOfNonDominated": len(paths),
            "steps": list(steps),
            "shiftedWeight": list(shiftedWeight),
            "paths": paths,
            "time": self.time
        }
        frame = pd.DataFrame.from_dict(log_obj, orient='index')
        frame = frame.transpose()
        frame.to_csv("./log/results.csv", mode='a', index = False, header=False)

    def logAllGenerationalSteps(self, objectiveTuple, paths, generation):
        #Check if csv for generations exist
        if not os.path.exists(self.logPath +"/log.csv"):
            with open(self.logPath +"/log.csv", "w") as f:
                header = "generation, map, width, height, algorithm, crossover, mutation, popsize, n_eval, samplingFunction, repairFunction, shiftingMethod, seed, objectiveValues, paths\n"
                f.write(header)
                f.close()
        formattedPaths = []
        for x in paths:
            formattedPaths.append(x[0])

        log_obj = {
            "generation" : generation,
            "map": self.map,
            "width": self.width,
            "height": self.height,
            "algorithm": self.algorithm,
            "crossover": self.crossover,
            "mutation": self.mutation,
            "popsize": self.popsize,
            "n_eval": self.n_eval,
            "samplingFunction": self.samplingFunction,
            "repairFunction": self.repairFunction,
            "shiftingMethod": self.shiftingMethod,
            "seed": self.seed,
            "objectives": list(objectiveTuple),
            "paths": formattedPaths,
        }
        frame = pd.DataFrame.from_dict(log_obj, orient='index')
        frame = frame.transpose()
        frame.to_csv(self.logPath +"/log.csv", mode='a', index = False, header=False)

    def logOptGenerationalSteps(self, objectiveTuple, paths, generation):
        #Check if csv for generations exist
        if not os.path.exists(self.logPath +"/optLog.csv"):
            with open(self.logPath +"/optLog.csv", "w") as f:
                header = "generation, map, width, height, algorithm, crossover, mutation, popsize, n_eval, samplingFunction, repairFunction, shiftingMethod, seed, objectiveValues, paths\n"
                f.write(header)
                f.close()
        formattedPaths = []
        for x in paths:
            formattedPaths.append(x[0])

        log_obj = {
            "generation" : generation,
            "map": self.map,
            "width": self.width,
            "height": self.height,
            "algorithm": self.algorithm,
            "crossover": self.crossover,
            "mutation": self.mutation,
            "popsize": self.popsize,
            "n_eval": self.n_eval,
            "samplingFunction": self.samplingFunction,
            "repairFunction": self.repairFunction,
            "shiftingMethod": self.shiftingMethod,
            "seed": self.seed,
            "objectives": list(objectiveTuple),
            "paths": formattedPaths,
        }
        frame = pd.DataFrame.from_dict(log_obj, orient='index')
        frame = frame.transpose()
        frame.to_csv(self.logPath +"/optLog.csv", mode='a', index = False, header=False)

    def log_best_paths_as_pickle(self, object: dict, filename: str) -> None:
        """Saves the best paths to a pickle file named appropriately.

        Args:
            object (dict): Best paths and objective values as dict
            filename (str): Chosen filename
        """
        with open(f"./pickle_objects/{filename}", "wb") as f:
            pickle.dump(object, f)
            f.close()