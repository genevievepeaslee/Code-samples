import sys
from collections import defaultdict
import scipy.sparse
import scipy.spatial
import sklearn.metrics.pairwise
from collections import Counter

'''
COMMAND LINE ARGUMENTS
<training data> <test data> <k value> <similarity func>  
training and test data: .txt files
k value: int > 0
similarity func: 0 or 1
'''

def classify(train_matrix, test_matrix, distance_metric, k_val, trainIDs_true):
    """
    Use training vectors to classify test data using kNN. Measure distance using either cosine similarity or Euclidean
    distance between vectors.
    :param train_matrix: training vectors in sparse matrix form
    :param test_matrix: test vectors in sparse matrix form
    :param distance_metric: int: 0 for
    :param k_val: number of closest neighbors whose "votes" are counted when classifying a new instance
    :param trainIDs_true: true class of each training instance in order of instance IDs
    :return: dict of test instances organized by predicted class (class --> {inst, inst, inst...})
    """
    instances_per_predictedclass = defaultdict(set)

    # distance value for each test vector to each training vector
    all_distances = sklearn.metrics.pairwise.pairwise_distances(test_matrix, train_matrix, metric=distance_metric)

    for inst_id in range(test_matrix.shape[0]):
        this_id_distances = all_distances[inst_id]
        votes = Counter()  # class --> votes
        # for cla in classes:  # initialize with each class as a key --> 0
        #     votes[cla] = 0

        # find k most similar (closest) training instances
        top_k_indexes = list(this_id_distances.argsort())[:k_val]  # indexes of closest vectors
        for index in top_k_indexes:
            votes[trainIDs_true[index]] += 1  # increment vote for class of each of k-closest neighbors
        probs = []  # list of tuples: (P(class|x), class)
        for c in votes:
            probs.append(tuple([votes[c] / k_val, c]))
        probs.sort(reverse=True)

        # add instance id to most likely class
        instances_per_predictedclass[probs[0][1]].add(inst_id)  # probs[0][1] is most likely class

        # # print results
        # with open(sys.argv[5], 'w', newline="") as sys_file:
        #     sys_file.write("IDnum" + str(inst_id) + " " + trainIDs_true[inst_id] + " ")
        #     for item in probs:
        #         sys_file.write(item[1] + " {:.5f}\t".format(item[0]))
        #     sys_file.write("\n")

    return instances_per_predictedclass


def confusion_matrix(true, predicted):
    """
    Print to stdout a confusion matrix and average accuracy.
    :param true: instances organized by true class (dict: class --> {inst, inst, inst})
    :param predicted: instances organized by predicted class (dict: class --> {inst, inst, inst}
    :return: tuple containing number of correctly classified instances and total number of instances: (right, total)
    """
    classes = list(true.keys())
    for i in range(len(classes)):
        print("\t" + classes[i], end="")
    print()
    right = 0
    total = 0
    for j in range(len(classes)):
        print(classes[j] + "\t", end="")
        for k in range(len(classes)):
            value = len((predicted[classes[k]] & true[classes[j]]))
            total += value
            if j == k:
                right += value
            print(str(value) + "\t", end="")
        print()

    return right, total


def make_matrix(vectors, feature_map):
    """
    Convert a dict representation of feature vectors into a sparse matrix representation.
    :param vectors: data in dict format: instance_id --> {feat:count, feat:count, ...}
    :param feature_map: dict mapping each feature to a unique int
    :return: scipy sparse matrix (csr) version of <vectors> (rows are instances, columns correspond to features)
    """
    # make matrix: rows are instances, column indexes correspond to feature indexes
    matrix = scipy.sparse.dok_matrix((len(vectors), len(feature_map)), dtype=float)
    for inst_id, feat_counts in vectors.items():
        for feat in feat_counts:
            if feat in feature_map:
                index_of_feat = feature_map[feat]
                matrix[inst_id, index_of_feat] = feat_counts[feat]
    # convert to csr_matrix
    return matrix.asformat('csr')


def main():
    """
    Given labeled train and test data, classify both sets of data using k-nearest neighbor and print accuracies
    compared to ground truth labels.

    Data is .txt format with one document/instance per line in the form:
        <true_label> <feat1>:<value> <feat2>:<value> <feat3>:<value>...
        e.g. talk.politics.guns a:11 about:2 absurd:1 again:1 an:1 ...

    """
    k_val = int(sys.argv[3])
    sim_function = int(sys.argv[4])
    if sim_function != 1 and sim_function != 2:
        raise ValueError("<sim function> parameter must be 1 (for Euclidean) or 2 (for cosine)")
    distance_metric = 'euclidean' if int(sys.argv[4]) == 1 else 'cosine'

    # Process training data
    features_to_ints = {}  # maps each feature to a unique int
    trainIDs_by_trueclass = defaultdict(set)  # set of training IDs belonging to each class
    train_trueclass_list = []  # list of true classes; index corresponds to trainID
    train_vectors = defaultdict(dict)  # vectors: instance_id --> {feat --> count, feat --> count}

    trainID_num = 0
    feat_counter = 0

    with open(sys.argv[1], 'r') as training_file:
        for line in training_file:
            split = line.split()
            trainIDs_by_trueclass[split[0]].add(trainID_num)  # add ID to true class set
            train_trueclass_list.append(split[0])  # put true class in list at index = trainID
            for feat in split[1:]:
                word = feat.split(":")[0]
                count = float(feat.split(":")[1])
                train_vectors[trainID_num][word] = count
                if word not in features_to_ints:
                    features_to_ints[word] = feat_counter
                    feat_counter += 1
            trainID_num += 1

    train_matrix = make_matrix(train_vectors, features_to_ints)
    train_predictions = classify(train_matrix, train_matrix, distance_metric, k_val, train_trueclass_list)

    # Process test data
    test_vectors = defaultdict(dict)  # store vectors (id_num --> {feat --> count, feat --> count}
    testIDs_by_trueclass = defaultdict(set)
    test_true_list = []
    testID_num = 0

    with open(sys.argv[2], 'r') as test_file:
        for line in test_file:
            spl = line.split()
            testIDs_by_trueclass[spl[0]].add(testID_num)  # add to set of ids in true class
            test_true_list.append(spl[0])  # add true class to list at index testID_num
            for f in spl[1:]:
                w = f.split(":")[0]
                c = f.split(":")[1]
                test_vectors[testID_num][w] = c
            testID_num += 1

    test_matrix = make_matrix(test_vectors, features_to_ints)

    test_predictions = classify(train_matrix, test_matrix, distance_metric, k_val, train_trueclass_list)

    # print results
    print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n")
    result = confusion_matrix(trainIDs_by_trueclass, train_predictions)
    print("\nTraining accuracy={:.5f}".format(result[0] / result[1]))

    print("\nConfusion matrix for the test data:\nrow is the truth, column is the system output\n")
    result = confusion_matrix(testIDs_by_trueclass, test_predictions)
    print("\nTest accuracy={:.5f}".format(result[0] / result[1]))


if __name__ == "__main__":
    main()

