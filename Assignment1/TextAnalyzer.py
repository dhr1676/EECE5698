import sys
import argparse
import numpy as np
from pyspark import SparkContext


def toLowerCase(s):
    """ Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()


def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])


def get_pair(s):
    """
    This function is to transfer a String variable into a tuple variable.
    The input of this function is a String variable like this:
    str("'string_word', int or float number").
    And the return of this function is a tuple variable like this:
    tuple("string_word", int or float number)
    Such tuple represents the key-value pair of the RDD of the form (WORD,VAL).

    :param s: The input of this function is a String type variable,
            which represents one line of the RDD.
    :return: The return of the function is a tuple pair of the form (WORD,VAL),
            where the first element of the tuple is a String type word,
            and the second element of the tuple is an int or float type number.
    """
    w1 = s.replace("\'", "")            # Remove the single quote of the input string
    w2 = w1.replace("(", "")            # Remove the left bracket of the input string
    w3 = w2.replace(")", "")            # Remove the right bracket of the input string
    w_list = w3.split(", ")             # Separate the input string by a comma and space
    return w_list[0], eval(w_list[1])   # Return the tuple of the form (WORD,VAL)


def clean_text(_word_list):
    """
    This function is to remove non alphabetic characters in the input word list,
    and return a non-repeating word list.

    :param _word_list: The input of the function is List type variable,
            which each element of the list is a word may be a non-alphabetic word.
    :return: The return of the function is a List type variable,
            which contains only non-repeating alphabetic words
    """
    clean_content_set = set()   # Declare a Set type variable to store result and remove duplication.
    for word in _word_list:
        clean_word = stripNonAlpha(word)        # Remove non alphabetic characters
        clean_content_set.add(clean_word)       # Add the cleaned word into the set.
    if "" in clean_content_set:
        clean_content_set.remove("")            # Remove the empty string
    return list(clean_content_set)              # Return the result and transfer the set into a list type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Analysis through TFIDF computation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', help='Mode of operation', choices=['TF', 'IDF', 'TFIDF', 'SIM', 'TOP'])
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master', default="local[20]", help="Spark Master")
    parser.add_argument('--idfvalues', type=str, default="idf",
                        help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other', type=str, help='Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Text Analysis')

    if args.mode == 'TF':
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed

        file_path = args.input                          # Read the input file path by using the args
        raw_text = sc.textFile(file_path)
        w1 = raw_text.map(toLowerCase)                  # Transfer the string into lower case
        w2 = w1.flatMap(lambda x: x.split())            # Separate the string by the space, "\n" and "\t" and flapMap
        w3 = w2.map(stripNonAlpha)                      # Remove the non-alphabetic characters
        w4 = w3.map(lambda word: word.encode('utf-8'))  # Transfer the Unicode word into a string type
        w5 = w4.filter(lambda x: x != "")               # Remove the empty string
        w6 = w5.map(lambda word: (word, 1))             # Form a key-value pair to count the number of each word
        w7 = w6.reduceByKey(lambda x, y: x + y)         # Use reduceByKey to sum
        word_list = w7.sortBy(lambda x: - x[1])         # Sort the result by descending

        output_path = args.output                       # Set output saving path
        word_list.saveAsTextFile(output_path)           # Save to text file

    if args.mode == 'TOP':
        # Read file at args.input, comprising strings representing pairs of the form (TERM,VAL),
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        file_path = args.input                          # Read the input file path by using the args
        raw_text = sc.textFile(file_path,
                               use_unicode=False)       # Read the input text file as str type
        w1 = raw_text.map(get_pair)                     # Use function to extract
        w2 = w1.takeOrdered(20, lambda x: -x[1])        # Let the RDD sort by descending and take the first 20 elements
        top_rdd = sc.parallelize(w2)                    # Transfer the list type into RDD type
        output_path = args.output                       # Set output saving path
        top_rdd.saveAsTextFile(output_path)             # Save to text file

    if args.mode == 'IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings "" are removed
        file_path = args.input
        raw_text = sc.wholeTextFiles(file_path)
        corpus_num = raw_text.count()
        w1 = raw_text.map(lambda x: x[1])
        w2 = w1.map(toLowerCase) \
            .map(lambda x: x.split()) \
            .map(clean_text) \
            .flatMap(lambda x: x).map(lambda word: word.encode('utf-8')) \
            .map(lambda word: (word, 1)) \
            .reduceByKey(lambda x, y: x + y) \
            .map(lambda x: (x[0], (np.log(float(float(corpus_num) / x[1]))))) \
            # .sortBy(lambda x: -x[1])
        output_path = args.output
        w2.saveAsTextFile(output_path)

    if args.mode == 'TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value.

        tf_scores_path = args.input
        idf_scores_path = args.idfvalues
        tf_scores = sc.textFile(tf_scores_path, use_unicode=False)
        idf_scores = sc.textFile(idf_scores_path, use_unicode=False)
        idf_dict = idf_scores \
            .map(get_pair) \
            .collectAsMap()
        b_idf_dict = sc.broadcast(idf_dict)
        tfidf_list = tf_scores \
            .map(get_pair) \
            .map(lambda pair: (pair[0], b_idf_dict.value[pair[0]] * pair[1])) \
            .sortBy(lambda x: -x[1])
        output_path = args.output
        tfidf_list.saveAsTextFile(output_path)

    if args.mode == 'SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output.
        # Both input files contain strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase, letter-only string and VAL is a numeric value.
        tfidf_1_scores_path = args.input
        tfidf_2_scores_path = args.other
        tfidf_1_scores = sc.textFile(tfidf_1_scores_path, use_unicode=False)
        tfidf_2_scores = sc.textFile(tfidf_2_scores_path, use_unicode=False)
        tfidf_1_scores_dict = tfidf_1_scores \
            .map(get_pair) \
            .collectAsMap()
        tfidf_2_scores_dict = tfidf_2_scores \
            .map(get_pair) \
            .collectAsMap()
        cross_set = set([x for x in tfidf_1_scores_dict if x in tfidf_2_scores_dict])
        t1_square, t2_square, numerator = 0.0, 0.0, 0.0
        for word in cross_set:
            numerator += float(tfidf_1_scores_dict[word]) * float(tfidf_2_scores_dict[word])
        for word in tfidf_1_scores_dict:
            t1_square += float(tfidf_1_scores_dict[word]) ** 2
        for word in tfidf_2_scores_dict:
            t2_square += float(tfidf_2_scores_dict[word]) ** 2
        cos = numerator / np.sqrt(t1_square * t2_square)
        output = "sim result of " + tfidf_1_scores_path + " and " + tfidf_2_scores_path + " is " + str(cos)
        result = sc.parallelize([output])
        print output
        output_path = args.output
        result.saveAsTextFile(output_path)
