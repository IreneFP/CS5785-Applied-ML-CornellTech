from nltk.corpus import stopwords

stopWords = stopwords.words('english')
isStopWord = lambda w: w in stopWords or len(w) == 1


def getAllReviews():
    filenames = ["yelp_labelled.txt", "amazon_cells_labelled.txt", "imdb_labelled.txt"]
    reviews = []

    for file in filenames:
        with open(file, 'r') as labelFile:
            reviews += labelFile.readlines()

    return reviews


def cleanUpReviews(data):
    clean = []
    for review in data:
        temp = review.split('\t')
        clean += [(temp[0].rstrip(), temp[1].strip())]
    return clean


def getLabeledReviews():
    labelFreq = {0: [], 1: []}

    for review in getAllReviews():
        label = int(review.split()[-1])
        labelFreq[label].append(review.split('\t')[0].rstrip())

    return labelFreq[0], labelFreq[1]


# 1.a
def areLabelsBalanced():
    labeledReviews = getLabeledReviews()
    if len(labeledReviews[0]) == len(labeledReviews[1]):
        print("Labels are balanced. Each is represented {}".format(len(labeledReviews[0])))
        return True
    else:
        print("Labels are not balanced. 0/1 ratio = {}".format(len(labeledReviews[0]) / len(labeledReviews[1])))
    return False


# 1.b
def preprocessData(data):
    reviews = []

    """
    Remove stop words because they contribute nothing to the sentiment. 
    They are just noise. Do it first to make other preprocessing quicker.
    """
    for d in data:
        noStop = ' '.join(word for word in d.split() if not isStopWord(word))
        reviews += [noStop]

    """
    Analyzing if words are upper/lower case is more for analyzing the intensity of the sentiment rather than classifying it. 
    """
    reviews = list(map(lambda r: r.lower(), reviews))

    """
    Punctuations will most likely be noise or add to the intensity of the sentiment 
    but is not a deciding factor for the classification of the sentiment.
    """
    puncts = [",", ".", "!", "$", ":", ";", "_", "/", "%", "(", ")", "+", "-", "#", "\"", "\'"]
    puncts.extend(str(i) for i in range(10))

    for p in puncts:
        reviews = list(map(lambda s: s.replace(p, ""), reviews))

    """
    Lemmatization
    """
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    for index, review in enumerate(reviews):
        reviews[index] = " ".join(list(map(lambda word: ps.stem(word), review.split())))

    return reviews


# 1.c
def splitData():
    label0, label1 = getLabeledReviews()
    train, test = [], []

    for i in range(0, len(label0), 500):
        stop1, stop2 = i + 400, i + 500
        train += label0[i: stop1] + label1[i: stop1]
        test += label0[stop1: stop2] + label1[stop1: stop2]

    return train, test


# 1.d
def makeBOW():
    from collections import defaultdict

    train, test = splitData()
    bagTrain, bagTest = defaultdict(int), defaultdict(int)
    cleanTrain, cleanTest = preprocessData(train), preprocessData(test)

    """
    Can't go through testing data yet because we need to build the feature vectors out of the training data.
    If we did this with the testing data we would be overfitting.
    """
    for review in cleanTrain:
        for word in review.split():
            bagTrain[word] += 1

    """
    Since we are iterating through the testing data, if the word is not in the dictionary at this point it
    is not one of the feature vectors so we ignore its prescence in the review.
    """
    for review in test:
        for word in review:
            if word in bagTrain.keys():
                bagTest[word] += 1

    indexForWord = {word: index for index, word in enumerate(bagTrain.keys())}
    bowTrain, bowTest = [], []
    uniqueWords = indexForWord.keys()
    amtWords = len(uniqueWords)

    for review in cleanTrain:
        features = [0] * amtWords
        for word in review.split():
            features[indexForWord[word]] += 1
        bowTrain.append(features)

    for review in cleanTest:
        features = [0] * amtWords
        for word in review.split():
            if word in uniqueWords:
                features[indexForWord[word]] += 1
        bowTest.append(features)

    return bowTrain, bowTest


# 1.e
def postprocessData(bag):
    from numpy.linalg import norm

    absTrain, absTest = 0, 0
    
    for review in bag[0]:
        for freq in review:
            absTrain += abs(freq)

    for review in bag[1]:
        for freq in review:
            absTest += abs(freq)
    
    """
    l1 normalization because it works best with sparse data 
    by not accouting weight for zero-values
    """

    normTrain = [list(map(lambda feature: feature/absTrain, review)) for review in bag[0]]
    normTest = [list(map(lambda feature: feature/absTest, review)) for review in bag[1]]


    return normTrain, normTest


def getTrainTestLabels():
    trainLabels, testLabels = [], []

    for _ in range(3):
        trainLabels += [0] * 400 + [1] * 400
        testLabels += [0] * 100 + [1] * 100

    return trainLabels, testLabels


def getFeatures():
    train, _ = splitData()
    cleanTrain = preprocessData(train)

    uniqueWords = []

    for review in cleanTrain:
        for word in review.split():
            if word not in uniqueWords:
                uniqueWords += [word]

    return uniqueWords


def getnGrams():
    train, _ = splitData()
    cleanTrain = preprocessData(train)

    uniqueGrams = []

    for review in cleanTrain:
        words = review.split()
        for index in range(len(words) - 1):
            grams = words[index] + " " + words[index + 1]
            if grams not in uniqueGrams:
                uniqueGrams += [grams]

    return uniqueGrams

# 1.f
def prediction(normBags, getFn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.naive_bayes import GaussianNB

    trainLabels, testLabels = getTrainTestLabels()
    trainNorm, testNorm = normBags[0], normBags[1]

    lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    lr.fit(trainNorm, trainLabels)

    predictions = lr.predict(testNorm)
    predScore = lr.score(testNorm, testLabels)
    confMatrix = confusion_matrix(testLabels, predictions)

    print("Prediction score: {}".format(predScore))
    print()
    print("Confusion Matrix:")
    print(confMatrix)

    weightVector = list(lr.coef_[0])
    maxWeight, minWeight = max(weightVector), min(weightVector)
    maxIndex, minIndex = weightVector.index(maxWeight), weightVector.index(minWeight)
    uniqueWords = getFn()
    wordForIndex = {index: word for index, word in enumerate(uniqueWords)}

    print()
    print("Most positive word: {}".format(wordForIndex[maxIndex]))
    print("Most negative word: {}".format(wordForIndex[minIndex]))
    print()

    bern = GaussianNB()
    bern.fit(trainNorm, trainLabels)
    bernPred = bern.predict(testNorm)
    bernScore = bern.score(testNorm, testLabels)
    bernConf = confusion_matrix(testLabels, bernPred)


    print("-" * 10 + "Gaussian" + "-" * 10)
    print("Score: {}".format(bernScore))
    print("Confusion Matrix:")
    print(bernConf)
    print()
    print()


# 1.g
def ngramModel():
    from collections import defaultdict

    train, test = splitData()
    bagTrain, bagTest = defaultdict(int), defaultdict(int)
    cleanTrain, cleanTest = preprocessData(train), preprocessData(test)

    for review in cleanTrain:
        words = review.split()
        for index in range(len(words) - 1):
            grams = words[index] + " " + words[index + 1]
            bagTrain[grams] += 1

    for review in cleanTest:
        words = review.split()
        for index in range(len(words) - 1):
            grams = words[index] + " " + words[index + 1]
            if grams in bagTrain.keys():
                bagTrain[grams] += 1

    indexForWord = {gram: index for index, gram in enumerate(bagTrain.keys())}
    bowTrain, bowTest = [], []
    uniqueGrams = indexForWord.keys()
    amtGrams = len(uniqueGrams)

    for review in cleanTrain:
        features = [0] * amtGrams
        words = review.split()
        for index in range(len(words) - 1):
            gram = words[index] + " " + words[index + 1]
            features[indexForWord[gram]] += 1
        bowTrain.append(features)

    for review in cleanTest:
        features = [0] * amtGrams
        words = review.split()
        for index in range(len(words) - 1):
            gram = words[index] + " " + words[index + 1]
            if gram in uniqueGrams:
                features[indexForWord[gram]] += 1
        bowTest.append(features)

    return bowTrain, bowTest


# 1.h
def pca():
    from numpy.linalg import svd
    from numpy import diag
    from numpy import where
    b = makeBOW()
    b = postprocessData(b)
    uTr, sTr, vTr = svd(b[0])
    uTe, sTe, vTe = svd(b[1])

    for r in [10, 50, 100]:
        print("Predictions for {} dimensions: ".format(r))
        print()
        trNorm = uTr[:, :r] @ diag(sTr)[:r, :r] @ vTr[:r, :]
        teNorm = uTe[:, :r] @ diag(sTe)[:r, :r] @ vTe[:r, :]
        try:
            prediction([trNorm, teNorm], getFeatures)
        except:
            print(r)
        print()

    print("*" * 25)
    print(" " * 8 + "NGRAM")

    b = ngramModel()
    postprocessData(b)
    uTr, sTr, vTr = svd(b[0])
    uTe, sTe, vTe = svd(b[1])

    for r in [10, 50, 100]:
        print("Predictions for {} dimensions: ".format(r))
        print()
        trNorm = uTr[:, :r] @ diag(sTr)[:r, :r] @ vTr[:r, :]
        teNorm = uTe[:, :r] @ diag(sTe)[:r, :r] @ vTe[:r, :]
        prediction([trNorm, teNorm], getnGrams)
        print()


if __name__ == "__main__":
    # splitData()
    # preprocessData()
    # b = makeBOW()
    # postprocessData(b)
    # prediction(b, getFeatures)
    # prediction(b, .txt)

    # print()
    # print("*" * 30)

    # b = ngramModel()
    # postprocessData(b)
    # prediction(b, getnGrams)

    pca()
    
    # train, test = ngramModel()
    # print(train)

    #print(len(getAllReviews()))
