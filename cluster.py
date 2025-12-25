import main as m
import os
import random
from multiprocessing import Process, Pool, freeze_support, get_context
from copy import deepcopy

def main():

    os.mkdir("opt_log_second")
    os.mkdir("opt_log")
    os.mkdir("./pickle_objects")
    os.mkdir("./pickle_objects_2")
    os.mkdir("./all_log")
    os.mkdir("./all_log_second")

    maps = [0,2,3]
    width = 50
    height = 50
    algorithms = 5
    crossovers = 2
    mutations = 0
    pop = 100
    total_n_eval = 100000
    eval_ratios = [0.3, 0.5, 0.7] # This tells us how much of the total evaluations the first run gets
    shiftingMethods = [1]
    seeds = [42, 69, 420, 1080, 1337, 617991, 799403, 302116, 414881, 718149, 659294, 327967, 4978, 167867, 247737, 890651, 853402, 996794, 489263, 972757, 269475, 282126, 397562, 400459, 353156, 202975, 684799, 190391, 591868, 296699, 856797]
    number_of_opt_solutions = ["all"]
    combinations_run1 = []
    combinations_run2 = []
    
    #getCombinations(maps, width, height, algorithms, crossovers, mutations, pop, total_n_eval, eval_ratios, number_of_opt_solutions, shiftingMethods, seeds)
    
    # We need to test different ratios for the n_evals in both runs
    base_configs = [
        [0, width, height, algorithms, crossovers, mutations, pop, total_n_eval, 1],
        [2, width, height, algorithms, crossovers, mutations, pop, total_n_eval, 1],
        [3, width, height, algorithms, crossovers, mutations, pop, total_n_eval, 5]
    ] # last part is shifting method and after we have all base configs we add the rest via cool array indexing

    conf_without_ratios = []

    for config in base_configs: # Modify all configs
        for seed in seeds:
            for ratio in eval_ratios:
                conf_with_seed = config + [seed] + [ratio] + ["all"]
                conf_without_ratios.append(conf_with_seed)
    
    combinations_run1 = []
    combinations_run2 = []

    for config in conf_without_ratios:
        ratio = config[-2]
        # Total eval index is at 7
        run1_comb = deepcopy(config)
        run1_comb[7] = int(run1_comb[7] * ratio)

        run2_comb = deepcopy(config)
        run2_comb[7] = int(run2_comb[7] * (1-ratio))

        combinations_run1.append(run1_comb)
        combinations_run2.append(run2_comb)

    args = []
    for combination in combinations_run1:
        args.append((combination, False))

    callMultiprocessing(args)

    # Second run with results of first run here
    args = []
    for combination in combinations_run2:
        args.append((combination, True))
    
    print("Starting second run")
    callMultiprocessing(args)

def callMultiprocessing(args: tuple):
    number_of_processes = min(100, os.cpu_count())
    with get_context("spawn").Pool(number_of_processes) as pool:
        pool.map(multiProcessSimulations, args)
        pool.close()

def getCombinations(maps, width, height, algorithms, crossovers, mutations, pop, total_eval, eval_ratios, number_of_opt_solutions, shiftingMethods, seeds) -> list:
    combinations_run1 = []
    combinations_run2 = []
    for map in maps:
        for algorithm in algorithms:
            for crossover in crossovers:
                for mutation in mutations:
                    for shiftingMethod in shiftingMethods:
                        for seed in seeds:
                            for ratio in eval_ratios:
                                for opt_solutions in number_of_opt_solutions:
                                    combinations_run1.append([map, width, height, algorithm, crossover, mutation, pop, int(ratio*total_eval), shiftingMethod, seed, ratio, opt_solutions])
                                    combinations_run2.append([map, width, height, algorithm, crossover, mutation, pop, int((1-ratio)*total_eval), shiftingMethod, seed, ratio, opt_solutions])


    return (combinations_run1, combinations_run2)

def multiProcessSimulations(args: tuple):
    c, second_run = args
    m.simulation(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], second_run)

if __name__ == "__main__":
    freeze_support()
    main()