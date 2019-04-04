import sys, argparse
from pyspark import SparkContext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PageRank Algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph',
                        help='Input Graph. The input should be a file containing one edge per line, '
                             'with each edge represented as a tuple of the form: (from_node,to_node)')
    parser.add_argument('--output', default="output", help='File in which PageRank scores are stored')
    parser.add_argument('--master', default="local[20]", help="Spark Master")
    parser.add_argument('--N', type=int, default=20, help='Level of Parallelism')
    parser.add_argument('--gamma', type=float, default=0.15, help='Interpolation parameter')
    parser.add_argument('--max_iterations', type=int, default=50, help='Maximum number of Iterations')
    parser.add_argument('--eps', type=float, default=0.01, help='Desired accuracy/epsilon value')
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Page Rank')

    lines = sc.textFile(args.graph)
    # Count: 600234
    # Content: [(p1, p2), (p1, p3), (p2, p4) ...]

    # Read graph and generate rdd containing node and outgoing edges 
    # from file containing edges
    graph_rdd = lines.map(eval) \
        .groupByKey() \
        .mapValues(list) \
        .partitionBy(args.N) \
        .cache()
    # Count: 1997
    # Content: [(p1, [p2, p3...]), (p2, [p3, p4...])]

    # Discover all nodes; this finds node with no outgoing edges as well	
    nodes = graph_rdd.flatMap(lambda (i, edgelist): edgelist + [i]) \
        .distinct() \
        .cache()
    # Count: 2000
    # Content: [p1, p2, p3...]

    # Initialize scores
    size = nodes.count()
    scores = nodes.map(lambda i: (i, 1.0 / size)).partitionBy(args.N).cache()

    # Main iterations
    i = 0
    err = args.eps + 1.0
    while i < args.max_iterations and err > args.eps:
        i += 1
        old_scores = scores
        joined = graph_rdd.join(scores)
        # Count: 1997
        # Content: [(p1, ([p2, p3...], 0.0005))]

        # scores = joined.values() \
        #     .flatMap(lambda (neighborlist, score): [(x, 1.0 * score / len(neighborlist)) for x in neighborlist]) \
        #     .reduceByKey(lambda x, y: x + y, numPartitions=args.N) \
        #     .mapValues(lambda x: (1 - args.gamma) * x + args.gamma * 1 / size) \
        #     .cache()

        scores = joined.values() \
            .flatMap(lambda (neighborlist, score): [(x, 1.0 * score / len(neighborlist)) for x in neighborlist]) \
            .reduceByKey(lambda x, y: x + y, numPartitions=20) \
            .mapValues(lambda x: (1 - 0.15) * x + 0.15 * 1 / size) \
            .cache()

        err = old_scores.join(scores).values() \
            .map(lambda (old_val, new_val): abs(old_val - new_val)) \
            .reduce(lambda x, y: x + y)

        old_scores.unpersist()
        print '### Iteration:', i, '\terror:', err

        # Give score to nodes having no incoming edges. All such nodes
    # should get score gamma / size
    remaining_nodes = nodes.map(lambda x: (x, args.gamma / size)).subtractByKey(scores)
    scores = scores.union(remaining_nodes)

    scores.sortBy(lambda (key, val): -val).saveAsTextFile(args.output)
