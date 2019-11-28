'''
Train a Naive Bayes document classification model (multivariate Bernoulli) from a corpus and classify test documents

args[1] training_data
args[2] test_data
args[3] class_prior_delta
args[4] cond_prob_delta
args[5] model_file (file to write to)
args[6] sys_output (accuracy info)

Training and test data format: <true class of doc1> <word1:count> <word2:count> <word3:count>...\n
                               <true class of doc2> <word2:count> <word2:count> <word3:count>...\n

Class_prior_delta is the smoothing factor for the class prior probabilities.
P(class_i) = (num docs in class_i + class_prior_delta) / (total docs + (number of classes)*(class_prior_delta))

Cond_prob_delta is the smoothing factor for the conditional probabilities (word given class)
P(word_i|class_i) = (num docs containing word_i in class_i + cond_prob_delta) / (num docs in class_i + (2 * cond_prob_delta))
'''


from collections import defaultdict
import math
import sys

class_prior_delta = float(sys.argv[3])
cond_prob_delta = float(sys.argv[4])

word_to_trainingIDs = defaultdict(set)  # word --> {trainingID, trainingID, trainingID}
trainingID_to_words = defaultdict(set)  # trainingID --> {word, word, word}
true_class_to_trainingIDs = defaultdict(set)  # category --> {trainingID, trainingID, trainingID}
training_true_class = []  # training_true_class[i] equals actual category of i'th training example
class_probs = dict()  # category --> log prob;
wordgivenclass_probs = defaultdict(dict)  # word --> {class:prob, class:prob, class:prob}


# Get training data
trainingID_num = -1

with open(sys.argv[1], 'r') as training_file:
    for line in training_file:
        trainingID_num += 1
        split = line.split()
        true_class_to_trainingIDs[split[0]].add(trainingID_num)
        training_true_class.append(split[0])  # index is trainingID_num
        for word in split[1:]:
            word = word[:word.rfind(':')]
            word_to_trainingIDs[word].add(trainingID_num)
            trainingID_to_words[trainingID_num].add(word)
# at this point, trainingID_num equals number of docs - 1

# calculate class prior probabilities (LOGS)
num_classes = len(true_class_to_trainingIDs.keys())
num_docs = trainingID_num + 1
for category in true_class_to_trainingIDs.keys():
    num_docs_in_class = len(true_class_to_trainingIDs[category])
    class_probs[category] = math.log((num_docs_in_class + class_prior_delta) / (num_docs + num_classes * class_prior_delta), 10)

classes = class_probs.keys()

# calculate word|class probabilities (NOT LOGS)
for w in word_to_trainingIDs.keys():  # all words in training feature vocab
    for c in classes:
        num_docs_in_class_with_w = len(true_class_to_trainingIDs[c] & word_to_trainingIDs[w])  # intersection of sets
        total_docs_in_class = len(true_class_to_trainingIDs[c])
        wordgivenclass_probs[w][c] = (num_docs_in_class_with_w + cond_prob_delta) / (total_docs_in_class + 2 * cond_prob_delta)

# calculate term3 for each class: sum of log(1 - P(word|class)) for every word in vocabulary
class_to_term3 = {}
for i in classes:
    total = 0
    for feature in word_to_trainingIDs.keys():
        total += math.log(1 - wordgivenclass_probs[feature][i], 10)
    class_to_term3[i] = total

# print model file
with open(sys.argv[5], 'w') as model_file:
    model_file.write("%%%%% prior prob P(c) %%%%%\n")
    for cl in classes:
        model_file.write(cl + "\t" + str(10**class_probs[cl]) + " " + str(class_probs[cl]) + "\n")
    model_file.write("%%%%% conditional prob P(f|c) %%%%%\n")
    for x in classes:
        model_file.write("%%%%% conditional prob P(f|c) c=" + x + " %%%%%\n")
        for word in sorted(wordgivenclass_probs.keys()):  # every word in the training feature vocabulary
            model_file.write(word + "\t" + x + "\t" + str(wordgivenclass_probs[word][x]) + " " +
                             str(math.log(wordgivenclass_probs[word][x], 10)) + "\n")

with open(sys.argv[6], 'a', newline="") as sys_output:
    sys_output.write("\n%%%%% training data:\n")

# classify training documents
# print results
# store final classification for accuracies
predicted_class_trainingIDs = defaultdict(set)  # class -> {ID, ID, ID}
for id in trainingID_to_words.keys():
    # calculate logP(document, class) for each class and store in joint_probs
    joint_probs = []
    max_joint_prob = -100000
    best_class = ""

    for clss in classes:
        term1 = class_probs[clss]
        term2 = 0
        for w in trainingID_to_words[id]:
            term2 += math.log(wordgivenclass_probs[w][clss] / (1 - wordgivenclass_probs[w][clss]), 10)
        thisjointprob = term1 + term2 + class_to_term3[clss]
        joint_probs.append(tuple([thisjointprob, clss]))
        # keep track of highest prob and associated class
        if thisjointprob > max_joint_prob:
            max_joint_prob = thisjointprob
            best_class = clss
    predicted_class_trainingIDs[best_class].add(id)

    # print true class and calculated prob of each possible class
    with open(sys.argv[6], 'a', newline="") as sys_output:
        sys_output.write("doc" + str(id) + " " + training_true_class[id])
        total = sum([item[0] for item in joint_probs])
        for item in sorted(joint_probs):
            sys_output.write(" " + item[1] + " " + str(item[0] / total) + " ")
        sys_output.write("\n")

with open(sys.argv[6], 'a', newline="") as sys_output:
    print("\n%%%%% test data:\n", file=sys_output)

true_class_to_testIDs = defaultdict(set)
testID_to_words = defaultdict(set)

test_true_class = []
predicted_class_testIDs = defaultdict(set)

testID_num = -1
# get test docs:
#   IDs --> words
#   true_class -> IDs
#   list of true_class at index <testID>
with open(sys.argv[2], 'r') as test_file:
    for line in test_file:
        testID_num += 1
        split = line.split()
        true_class = split[0]
        true_class_to_testIDs[true_class].add(testID_num)
        test_true_class.append(true_class)
        for word in split[1:]:
            word = word[:word.rfind(':')]
            testID_to_words[testID_num].add(word)

# results for test docs
for testID in testID_to_words.keys():
    # calculate logP(d,c) for each c and store in jointprobs
    jointprobs = []
    maxprob = -100000
    bestclass = ""

    for cls in classes:
        trm1 = class_probs[cls]
        trm2 = 0
        for feat in (testID_to_words[testID]):
            if feat not in word_to_trainingIDs.keys():
                trm2 += cond_prob_delta / (len(true_class_to_trainingIDs[cls]) + 2 * cond_prob_delta)
            else:
                trm2 += math.log(wordgivenclass_probs[feat][cls] / (1 - wordgivenclass_probs[feat][cls]), 10)
        thisjointprob = trm1 + trm2 + class_to_term3[cls]
        jointprobs.append(tuple([thisjointprob, cls]))
        if thisjointprob > maxprob:
            maxprob = thisjointprob
            bestclass = cls
    predicted_class_testIDs[bestclass].add(testID)

    # print results
    with open(sys.argv[6], 'a', newline="") as sys_output:
        sys_output.write("doc" + str(testID) + " " + test_true_class[testID])
        sm = 0
        for item in jointprobs:
            sm += item[0]
        for it in sorted(jointprobs):
            sys_output.write(" " + it[1] + " " + str(it[0] / sm) + " ")
        sys_output.write("\n")

# print accuracy info
classes = list(class_probs)
print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n")
for i in range(len(classes)):
    print("\t" + classes[i], end="")
print()
right = 0
all = 0
for j in range(len(classes)):
    print(classes[j] + "\t", end="")
    for k in range(len(classes)):
        value = len((predicted_class_trainingIDs[classes[k]] & true_class_to_trainingIDs[classes[j]]))
        all += value
        if j == k:
            right += value
        print(str(value) + "\t", end="")
    print()
print("\nTraining accuracy=" + str(right / all))

print("\nConfusion matrix for the test data:\nrow is the truth, column is the system output\n")
for l in range(len(classes)):
    print("\t" + classes[l], end="")
print()
r = 0
a = 0
for m in range(len(classes)):
    print(classes[m] + "\t", end="")
    for n in range(len(classes)):
        vl = len((predicted_class_testIDs[classes[n]] & true_class_to_testIDs[classes[m]]))
        a += vl
        if m == n:
            r += vl
        print(str(vl) + "\t", end="")
    print()
print("\nTest accuracy=" + str(r / a))