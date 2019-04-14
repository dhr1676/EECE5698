# -*- coding: utf-8 -*-
from time import time
import sys
import argparse
from pyspark import SparkContext
from operator import add
import numpy as np
from pyspark.mllib.random import RandomRDDs


def read_data(file_path, sparkContext):
    """
    Read data from file, and return RDD data
    :param file_path: data file path
    :param sparkContext: SparkContext
    :return: graph data RDD
    """
    data_rdd = sparkContext \
        .textFile(file_path) \
        .map(eval) \
        .map(lambda x: (x[0], x[1]))
    return data_rdd


def swap((u, v)):
    """
    Swap the elements of a pair tuple.
    """
    return (v, u)


def calc_degree(graph_rdd):
    """
    Calculate the degree for each node in the graph,
    return the degree result RDD
    :param graph_rdd:
    :return: degree RDD
    """
    all_degree = graph_rdd \
        .map(swap) \
        .union(graph_rdd) \
        .map(lambda (x, y): (x, 1)) \
        .reduceByKey(add, numPartitions=40)
    return all_degree


def calc_coloring(graph_rdd):
    """
    Calculate the color of the given graph
    :param graph_rdd:
    :return: color RDD
    """
    graph_rdd = graph_rdd \
        .map(swap) \
        .union(graph_rdd)
    nodes = graph_rdd.keys().distinct()

    color = nodes.map(lambda x: (x, 1))
    color_num = 1

    while True:
        graph_join_color = graph_rdd.join(color)
        neighbour = graph_join_color \
            .map(lambda (x, (a, bx)): (a, (x, bx))) \
            .groupByKey() \
            .map(lambda (x, y): (x, [n[1] for n in y]))
        color = neighbour.map(lambda (x, y): (x, hash(str(sorted(y)))))
        color_new = color \
            .map(swap) \
            .reduceByKey(add, numPartitions=40) \
            .map(lambda x: 1) \
            .reduce(add)
        if color_num != color_new:
            break
        color_num = color_new
    return color


def join2graph(graph1, graph2):
    graph_1 = graph1.map(swap) \
        .groupByKey() \
        .mapValues(list)
    graph_2 = graph2.map(swap) \
        .groupByKey() \
        .mapValues(list)
    graph = graph_1.join(graph_2).values()
    return graph


def is_isomorphic(graph1, graph2):
    # Get the n
    num_graph1 = graph1.count()
    num_graph2 = graph2.count()
    num_all = num_graph1 + num_graph2
    print "num all", num_all

    # Check whether 2 graphs have the same color distribution
    wl_graph1 = graph1.map(lambda (x, y): (x, 1)).reduceByKey(add)
    wl_graph2 = graph2.map(lambda (x, y): (x, 1)).reduceByKey(add)
    wl = wl_graph1.join(wl_graph2) \
        .values() \
        .filter(lambda (x, y): x != y) \
        .count()

    # Get the num of color
    num_color_graph1 = graph1.values().distinct().count()
    num_color_graph2 = graph2.values().distinct().count()
    num_color_all = num_color_graph1 + num_color_graph2
    print "num_color_all", num_color_all

    if wl == 0:
        if num_graph1 == num_graph2:
            if num_color_all < num_all:
                # If the two graphs have the same color distribution,
                # but the number of colors is less than n, then:
                print "The two graphs maybe isomorphic.\n"
            elif num_color_all == num_all:
                # If the two graphs have the same color distributions,
                # and the number of colors is exactly n, then:
                print "The two graphs are isomorphic.\n"
    else:
        # If the two graphs even don't have the same color distribution:
        print "The two graphs are not isomorphic.\n"

    joined = graph1.join(graph2)
    return joined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph Spark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', help='Mode of Operation', choices=['q1', 'q2', 'q3', 'q4'])
    parser.add_argument('input', default=None, help='Input Graph Data')
    parser.add_argument('--input2', default=None, help='The Second Input Graph Data')
    parser.add_argument('--output', default=None, help='Output Path')
    # parser.add_argument('--N', default=40, type=int, help='PartitionNum Level')
    parser.add_argument('--master', default="local[20]", help="Spark Master")

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    sc = SparkContext(args.master, 'Graph')

    if not args.verbose:
        sc.setLogLevel("ERROR")

    if args.mode == 'q1':
        graph_data = read_data(args.input, sc)
        graph_degree = calc_degree(graph_rdd=graph_data)
        graph_degree.saveAsTextFile(args.output)

    if args.mode == 'q2':
        graph_data = read_data(args.input, sc)
        color_rdd = calc_coloring(graph_rdd=graph_data)
        color_rdd.saveAsTextFile(args.output)

    if args.mode == 'q3' or args.mode == 'q4':
        graph_data_1 = read_data(args.input, sc)
        graph_data_2 = read_data(args.input2, sc)
        result = is_isomorphic(graph_data_1, graph_data_2)
        if args.output:
            result.saveAsTextFile(args.output)
