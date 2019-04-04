# -*- coding: utf-8 -*-
import numpy as np
import argparse
from time import time
from SparseVector import SparseVector
from LogisticRegression import readBeta, writeBeta, gradLogisticLoss, logisticLoss, lineSearch
from operator import add
from pyspark import SparkContext


def readDataRDD(input_file, spark_context):
    """
    Read data from an input file. Each line of the file contains tuples of the form
                (x,y)
     x is a dictionary of the form:
       { "feature1": value, "feature2":value, ...}
     and y is a binary value +1 or -1.
     The return value is an RDD containing tuples of the form
             (SparseVector(x),y)
    """
    return spark_context.textFile(input_file) \
        .map(eval) \
        .map(lambda (x, y): (SparseVector(x), y))


def getAllFeaturesRDD(dataRDD):
    """
    Get all the features present in grouped dataset dataRDD.
    The input is:
        - dataRDD containing pairs of the form (SparseVector(x),y).
    The return value is an RDD containing the union of
    all unique features present in sparse vectors inside dataRDD.
    """
    features = dataRDD \
        .map(lambda (x, y): x) \
        .map(lambda x: x.keys) \
        .flatMap() \
        .distinct()
    return features


def totalLossRDD(dataRDD, beta, lam=0.0):
    """
    Given dataRDD containing pairs of the form (SparseVector(x),y),
    sparse vector beta, compute the logistic loss
       l(β;x,y) = log( 1.0 + exp(-y * <β,x>) )
    The input is:
       - beta: a sparse vector β
       - dataRDD: (SparseVector(x),y)
       - lam: the regularization parameter λ
    """

    total_loss = dataRDD \
        .map(lambda (x, y): logisticLoss(beta, x, y)) \
        .reduce(lambda x, y: x + y)
    return total_loss + lam * beta.dot(beta)


def gradTotalLossRDD(dataRDD, beta, lam=0.0):
    """
    Given dataRDD containing pairs of the form (SparseVector(x),y),
    sparse vector beta, compute the gradient of regularized total logistic loss :
        ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β
    The input is:
       - beta: a sparse vector β
       - dataRDD: (SparseVector(x),y)
       - lam: the regularization parameter λ
    """
    grad_loss = dataRDD \
        .map(lambda (x, y): gradLogisticLoss(beta, x, y)) \
        .reduce(lambda x, y: x + y)
    l2_norm = 2 * lam * beta
    return grad_loss + l2_norm


def test(dataRDD, beta):
    """
    Output the quantities necessary to compute the accuracy, precision,
    and recall of the prediction of labels in a dataset under a given β.
    Inputs are:
         - data: an RDD containing pairs of the form (x,y)
         - beta: vector β
    The return values are
         - ACC, PRE, REC
    """
    ans = dataRDD \
        .map(lambda (x, y): (beta.dot(x), y)) \
        .cache()
    ans_P = ans.filter(lambda (x, y): x > 0)
    ans_N = ans.filter(lambda (x, y): x <= 0)
    TP = ans_P.filter(lambda (x, y): y == 1).count()
    FP = ans_P.filter(lambda (x, y): y == -1).count()
    TN = ans_N.filter(lambda (x, y): y == -1).count()
    FN = ans_N.filter(lambda (x, y): y == 1).count()
    P = ans.filter(lambda (x, y): x > 0).count()
    N = ans.filter(lambda (x, y): x <= 0).count()
    acc = 1.0 * (TP + TN) / (P + N)
    pre = 1.0 * TP / (TP + FP)
    rec = 1.0 * TP / (TP + FN)
    return acc, pre, rec


def train(dataRDD, beta_0, lam, max_iter, eps, test_data=None):
    start_time = time()  # Get the start time
    beta = beta_0  # Initialize the beta with input beta_0
    gradNorm = 2 * eps  # Initialize the norm by using input data
    k = 0  # Use a count k to record iterations

    while k < max_iter and gradNorm > eps:
        obj = totalLossRDD(dataRDD, beta, lam)
        grad = gradTotalLossRDD(dataRDD, beta, lam)
        gradNormSq = grad.dot(grad)
        gradNorm = np.sqrt(gradNormSq)

        fun = lambda x: totalLossRDD(dataRDD, x, lam)
        gamma = lineSearch(fun, beta, grad, obj, gradNormSq)

        beta = beta - gamma * grad
        if test_data is None:
            print 'k = ', k, '\tt = ', time() - start_time, '\tL(β_k) = ', \
                obj, '\t||∇L(β_k)||_2 = ', gradNorm, '\tgamma = ', gamma
        else:
            acc, pre, rec = test(test_data, beta)
            print 'k = ', k, '\tt = ', time() - start_time, '\tL(β_k) = ', \
                obj, '\t||∇L(β_k)||_2 = ', gradNorm, '\tgamma = ', \
                gamma, '\tACC = ', acc, '\tPRE = ', pre, '\tREC = ', rec
        k = k + 1

    return beta, gradNorm, k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Logistic Regression.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata', default=None,
                        help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata', default=None,
                        help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta',
                        help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float, default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')
    parser.add_argument('--N', type=int, default=2, help='Level of parallelism')

    # verbosity_group = parser.add_mutually_exclusive_group(required=False)
    # verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    # verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    # parser.set_defaults(verbose=False)

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Logistic Regression')

    print 'Reading training data from', args.traindata
    traindata = readDataRDD(args.traindata, sc)
    if args.testdata is not None:
        print 'Reading test data from', args.testdata
        testdata = readDataRDD(args.testdata, sc)
        # print 'Read',len(testdata),'data points with',len(getAllFeatures(testdata)),'features'
    else:
        testdata = None

    beta0 = SparseVector({})

    beta, gradNorm, k = train(dataRDD=traindata,
                              beta_0=beta0,
                              lam=args.lam,
                              max_iter=args.max_iter,
                              eps=args.eps,
                              test_data=testdata)

    print 'Algorithm ran for', k, 'iterations. Converged:', gradNorm < args.eps
    print 'Saving trained β in', args.beta
    writeBeta(args.beta, beta)
