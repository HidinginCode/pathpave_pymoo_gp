import os
import shutil
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import csv
import pickle

class Logger():
    def __init__(self, second_run: bool = False) -> None:
        """Init method that creates log directory if it does not exist."""
        self.second_run = second_run
        if not second_run:
            self.basePath = "./log"
            self.pickle_path = "./pickle_objects"

        else:
            self.basePath = "./log_2"
            self.pickle_path = "./pickle_objects_2"
    
    def createLogFile(self, map, width, height, algorithm, crossover, mutation, popsize, n_eval, samplingFunction, repairFunction, shiftingMethod, seed, eval_ratio, number_of_opt_solutions, totalTime):
        """Creates a logfile for the path."""
        eval_ratio_str = str(eval_ratio).replace(".", "_")
        self.logName = f"{map.name}_{width}_{height}_{algorithm.__class__.__name__}_{crossover.__class__.__name__}_{mutation.__class__.__name__}_{popsize}_{eval_ratio_str}_{number_of_opt_solutions}_{samplingFunction.__class__.__name__}_{repairFunction.__class__.__name__}_{shiftingMethod}_{seed}"
        self.logPath = self.basePath + "/" + self.logName

        # Set class variables
        self.map = map.name
        self.width = width
        self.height = height
        self.algorithm = algorithm.__class__.__name__
        self.crossover = crossover.__class__.__name__
        self.mutation = mutation.__class__.__name__
        self.popsize = popsize
        self.n_eval = n_eval
        self.eval_ratio = eval_ratio
        self.samplingFunction = samplingFunction.__class__.__name__
        self.repairFunction = repairFunction.__class__.__name__
        self.seed = seed
        self.time = totalTime
        self.number_of_opt_solutions = number_of_opt_solutions

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
        frame.to_csv(f"{self.basePath}/results.csv", mode='a', index = False, header=False)

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

    def logOptGenerationalSteps(self, objectiveTuple, paths, generation, second_run):
        if not second_run:
            base_path = "opt_log"
        else:
            base_path = "opt_log_second"

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
            "ratio": self.eval_ratio,
            "samplingFunction": self.samplingFunction,
            "repairFunction": self.repairFunction,
            "shiftingMethod": self.shiftingMethod,
            "seed": self.seed,
            "objectives": list(objectiveTuple),
            "paths": formattedPaths,
        }
        dir = f"{self.map}_{self.eval_ratio}_opt"
        os.makedirs(os.path.join(base_path, dir), exist_ok=True)
        with open(base_path+"/"+dir+"/"+f"/opt_{self.seed}_{generation}.pickle", "wb") as f:
            pickle.dump(log_obj, f)

    def log_best_paths_as_pickle(self, object: dict, filename: str) -> None:
        """Saves the best paths to a pickle file named appropriately.

        Args:
            object (dict): Best paths and objective values as dict
            filename (str): Chosen filename
        """
        with open(f"{self.pickle_path}/{filename}", "wb") as f:
            pickle.dump(object, f)
            f.close()