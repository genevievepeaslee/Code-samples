"""
POS tagger using a given a MaxEnt model and implementing beam search.

INPUTS
test_data = sys.argv[1] (one word and its vector of features and values per line)
    <word> <true_pos> <feat1> <val1> <feat2> <val2> ...
    1-0-The DT curW=The 1 prevW=BOS 1 prev2W=BOS 1 nextW=Arizona 1
boundary_file = sys.argv[2] (one number per line: number of words in each sentence in test_data)
model_file = sys.argv[3] (maxent model parameters)
    FEATURES FOR CLASS NNP
    <default> 3.7912278052488615
    curW=Pierre 1.0055824571891294
    prevW=BOS 0.15158438156724433
    ...
    FEATURES FOR CLASS NNS
     <default> 2.956903172545647
     curW=Pierre -0.050694168880812344
     ...
sys_output filename = sys.argv[4]
    Format is <word> <true_pos> <predicted_pos> <prob(predicted_pos | word)>
beam_size = sys.argv[5]
    At each time step prune paths whose prob is not within <beam_size> of prob of most probable path
topN = sys.argv[6]
    Explore only N most likely tags for each word
topK = sys.argv[7]
    Explore only K paths at each time step
"""

import sys
import re
from collections import defaultdict
from math import exp
import copy
import math
import numpy as np

def get_model_weights(file_name):
    """
    Read model file and store in a dictionary of dictionaries mapping each POS class to the features and their weights:
    {class : {feat:weight, feat:weight...}}
    Return dictionary and a list of the POS classes
    """
    weights = {}
    with open(file_name, 'r') as model_file:
        cla = ""
        for l in model_file:
            if l.startswith("FEATURES FOR CLASS"):
                cla = l.split()[-1]
                weights[cla] = {}
            else:
                feat_weight = l.split()
                weights[cla][feat_weight[0]] = float(feat_weight[1])
    return weights, list(weights.keys())


def get_top_n(classes_to_exponents, n, classes):
    """
    Calculate P(POS class | word) for each POS class and a given word using MaxEnt model and return top N classes
    :param classes_to_exponents: dictionary of {class --> summation of feature weights for given word}
    :param: n: max number of possible classes to keep per word
    :param: classes: all POS classes present in MaxEnt model file
    :return: a list of (prob, tag) tuples, sorted by probability in decreasing order, of length n
    """

    # calculate denominator: sum of all numerators
    Z = 0
    numerators = {}
    for cl, exponent in classes_to_exponents.items():
        numerators[cl] = exp(exponent)
        Z += numerators[cl]

    probs = []  # list of tuples (P(tag|word), tag)

    for c in classes:
        probs.append(tuple([numerators[c] / Z, c]))

    return sorted(probs, reverse=True)[:n]


def get_exponent_sums(list_of_features, model_weights, classes):
    """
    take a list of features, return a dictionary of feature weight summations per class
    :param: list_of_features:
    :param: model_weights:
    :param: classes: all POS classes present in MaxEnt model file
    """

    powers = {}
    for t in classes:
        pow = model_weights[t]['<default>']  # start the exponent-to-be at the default weight for the current class
        for feature in list_of_features:
            pow += model_weights[t].get(feature, 0)
        powers[t] = pow

    return powers


class Node:
    """
    Represents a node in the tree of possible tag sequences for a given sentence: one possible tag for a given word.
    """

    def __init__(self, word, tag, tag_prob, path_prob, parent):
        self.word = word
        self.tag = tag  # P(tag|word)
        self.tag_prob = tag_prob  # P(most probable preceding path)
        self.path_prob = path_prob
        self.parent = parent  # type Node

    def __str__(self):
        return "word=" + self.word + " tag=" + self.tag + " path_prob=" + str(self.path_prob)


def max_pathprob(node_list):
    max_p = 0
    for n in node_list:
        max_p = max(max_p, n.path_prob)
    return max_p


def main():
    beam_size = int(sys.argv[5])
    n = int(sys.argv[6])
    top_k = int(sys.argv[7])

    true_tag_by_id = []
    predicted_tag_by_id = []

    # get model weights and list of classes
    # ex.: {NNP: {"curW=Pierre":1.0055824571891294, "prevW=BOS":0.15158438156724433}}
    model_weights, classes = get_model_weights(sys.argv[3])

    # store list of test sentence lengths
    with open(sys.argv[2], 'r') as boundary_file:
        boundaries = [int(line) for line in boundary_file.readlines()]

    # clear sys_output file in case it's been written to before
    open(sys.argv[4], 'w')

    # PROCESS TEST DATA
    with open(sys.argv[1], 'r') as test_data:
        for sentence_length in boundaries:  # each sentence in test data
            # dict representation of tree of all possible tag sequences
            # key: word in the sentence
            # value: list of Nodes representing viable tags for word
            tree = defaultdict(list)

            # read in number of word vectors corresponding to sentence length in boundary file
            word_vectors = [test_data.readline() for index in range(sentence_length)]

            # process first word
            first_word_split = word_vectors[0].split(" ")
            first_word = first_word_split[0]
            # store true class of first word for later accuracy calculation
            true_tag_by_id.append(first_word_split[1])

            # get top_n for first word
            first_features = first_word_split[2::2]
            first_features.append("prevT=BOS")
            first_features.append("prevTwoTags=BOS+BOS")
            first_exponents = get_exponent_sums(first_features, model_weights, classes)
            first_top_n = get_top_n(first_exponents, n, classes)

            # make node for each top_n tag for first word
            BOS_node = Node(None, "BOS", 1, 1, None)  # parent node for each of first word's nodes
            first_nodes = []
            for tag_prob, tag in first_top_n:
                first_nodes.append(Node(first_word, tag, tag_prob, tag_prob, BOS_node))  # path_prob is the same as tag_prob for the first word

            # keep only paths with high enough probabilities
            log_max_prob = math.log(10, max_pathprob(first_nodes))  # max path prob to compare against
            for item in first_nodes:
                if math.log(10, item.path_prob) + beam_size >= log_max_prob:
                    tree[0].append(item)

            # continue with rest of words
            for word_position in range(1, sentence_length):
                curr_split = word_vectors[word_position].split(" ")
                curr_word = curr_split[0]

                # store true tag of current word
                true_tag_by_id.append(curr_split[1])

                # store features of current word
                curr_features = curr_split[2::2]

                # for maxent calculation
                # get weight summation per class for all features so far (same for all nodes, regardless of parent)
                curr_exponents = get_exponent_sums(curr_features, model_weights, classes)

                all_curr_nodes = []

                # for each possible parent node (node at the previous step),
                # generate previous_tag features and find top_n tags for this word;
                # make corresponding nodes
                max_path_prob = 0
                for parent_node in tree[word_position - 1]:
                    # form previous tag features
                    prev_tag = "prevT=" + parent_node.tag
                    if word_position == 1:
                        prev_two_tags = "prevTwoTags=BOS+" + parent_node.tag
                    else:
                        prev_two_tags = "prevTwoTags=" + parent_node.parent.tag + "+" + parent_node.tag

                    # finish exponent summation (add prev_tag features) per class (add to COPY of curr_exponents)
                    this_parent_exponents = copy.deepcopy(curr_exponents)
                    for clss in classes:
                        this_parent_exponents[clss] += model_weights[clss].get(prev_tag, 0)
                        this_parent_exponents[clss] += model_weights[clss].get(prev_two_tags, 0)

                    # get top_n tags for current word
                    curr_top_n = get_top_n(this_parent_exponents, n, classes)

                    # make nodes for each top_n tag
                    for tag_prob, tag in curr_top_n:
                        this_path_prob = tag_prob * parent_node.path_prob
                        max_path_prob = max(max_path_prob, this_path_prob)
                        all_curr_nodes.append(Node(curr_word, tag, tag_prob, this_path_prob, parent_node))

                # Prune: only keep top K, within beam size
                # beam
                log_max_prob = math.log(10, max_path_prob)
                nodes_to_keep = [node for node in all_curr_nodes if math.log(10, node.path_prob) + beam_size >= log_max_prob]

                # top_k
                nodes_to_keep = sorted(nodes_to_keep, key=lambda n: n.path_prob, reverse=True)[:top_k]

                # assign remaining nodes to tree at current word position
                tree[word_position] = nodes_to_keep

            if sentence_length > 1:
                best_final_node = max(tree[word_position], key=lambda n: n.path_prob)
            else:
                best_final_node = max(tree[0], key=lambda n: n.path_prob)

            # backtrace to get path that led to best_final_node
            sys_output_lines = []
            current_node = best_final_node
            predicted_tags_this_sent = []
            for i in range(0, sentence_length):
                sys_output_lines.append("{} {} {} {:.5f}\n".format(current_node.word, true_tag_by_id[-(i + 1)],
                                                                   current_node.tag, current_node.tag_prob))
                predicted_tags_this_sent.append(current_node.tag)
                current_node = current_node.parent
            with open(sys.argv[4], 'a') as sys_file:
                while sys_output_lines:
                    predicted_tag_by_id.append(predicted_tags_this_sent.pop())
                    sys_file.write(sys_output_lines.pop())

    # calculate and print accuracy across all sentences
    predicted_tag_by_id = np.array(predicted_tag_by_id)
    true_tag_by_id = np.array(true_tag_by_id)
    print(np.mean(predicted_tag_by_id == true_tag_by_id))

    # # DEBUGGING - print tree
    # for word_posit in range(len(tree)):
    #     for nod in tree[word_posit]:
    #         print(str(word_posit) + " " + str(nod))
    #         if nod.parent and nod.parent.word and nod.parent.tag:
    #             print("parent: " + nod.parent.word + " " + nod.parent.tag)
    #
    # print("predicted tags")
    # print(predicted_tag_by_id)


if __name__ == "__main__":
    main()

