import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


# return each row in csv file as a tuple in a list
def extractData(dataPath):
    rawData = pd.read_csv(dataPath, header=0)
    return [d for d in rawData.itertuples(index=False, name=None)]


def plotDigits(rawData):
    digitsSeen, subIndex = [], 1
    for dig in rawData:
        label, digit = dig[0], np.array(list(dig[1:]))

        if label not in digitsSeen:
            plt.subplot(2, 5, subIndex)
            plt.imshow(digit.reshape(28, 28))
            digitsSeen.append(label)
            subIndex += 1

        if subIndex == 10:
            break

    plt.savefig("digits.png")


def digitFrequency(rawData):
    labels = [d[0] for d in rawData]
    plt.hist(labels, bins=10, density=True)
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title('Digit Frequency')
    plt.xticks(list(range(10)))
    plt.savefig("Digit Frequency")


def getDistanceMatrix(rawData):
    dimensions = (len(rawData), len(rawData))
    distMatrix = [[99999 for _ in range(dimensions[0])] for _ in range(dimensions[0])]
    distMatrix = np.array(distMatrix)

    for index1, dig1 in enumerate(rawData):
        for index2, dig2 in enumerate(rawData):
            if index2 > index1:
                d1, d2 = np.array(dig1[1:]), np.array(dig2[1:])

                # euclidian distance
                distance = np.linalg.norm(d1 - d2)

                # dist(i,j)
                distMatrix[index1][index2] = distance
                distMatrix[index2][index1] = distance

    return distMatrix


# finds closest neighbor of first entry of every digit (0-9)
def findClosestNeighbor(rawData):
    matches = dict()
    distMatrix = getDistanceMatrix(rawData)

    i = 0
    # find closest neighbor
    for row in distMatrix:
        if not rawData[i][0] in matches.keys():
            minIndex = list(row).index((min(row)))
            matches[rawData[i][0]] = [rawData[i][1:], rawData[minIndex][1:]]

        if len(matches.keys()) == 10:
            break
        i += 1

    j = 1
    plt.figure()
    for i, v in matches.items():
        digit = np.array(v[0]).reshape(28, 28)
        match = np.array(v[1]).reshape(28, 28)

        plt.subplot(10, 2, j)
        plt.imshow(digit)

        plt.subplot(10, 2, j + 1)
        plt.imshow(match)

        j += 2

    plt.savefig("Best digit match")

    return matches


# compares all 0's and 1's
def binaryComparison(rawData, plot=False):
    zeros = np.array([d for d in rawData if d[0] == 0])
    ones = np.array([d for d in rawData if d[0] == 1])

    genuine = np.concatenate((cdist(ones, ones, 'euclidean').ravel(), cdist(zeros, zeros, 'euclidean').ravel()), axis=0)
    impostor = cdist(zeros, ones, 'euclidean').ravel()

    if plot is True:
        plt.figure()
        plt.hist(genuine, bins='auto', density=True, alpha=0.5, label="Genuine matches")
        plt.hist(impostor, bins='auto', density=True, alpha=0.5, label="Impostor matches")
        plt.legend()
        plt.show()

    return np.array(genuine), np.array(impostor)


def genROC(genuine, impostor):
    maxDist = max(impostor)
    amountG, amountI = len(genuine), len(impostor)
    truePosProb, falsePosProb = [], []
    eerRates = np.arange(0, 1.01, 0.02)

    for factor in np.arange(0, 1.01, 0.02):
        tempDist = maxDist * factor
        tempGen = len(np.where(genuine < tempDist)[0])
        tempImp = len(np.where(impostor < tempDist)[0])

        truePosProb += [tempGen / amountG]
        falsePosProb += [tempImp / amountI]

    plt.figure()
    plt.plot(falsePosProb, truePosProb, 'b-', label="ROC")
    plt.plot(eerRates, eerRates[::-1], '--r', label="EER")
    plt.title("ROC Curve")
    plt.xlabel("False Positives (%)")
    plt.ylabel("True Positives (%)")
    plt.grid()
    plt.legend()
    plt.savefig("ROC")


def knnClassifier(rawData, k=3):
    distMatrix = getDistanceMatrix(rawData)
    matches = []

    for index, row in enumerate(distMatrix):
        label1 = rawData[index][0]
        sortedRow = list(row)
        sortedRow.sort()
        tempDist = sortedRow[:k]

        indexes = [np.where(row == distance)[0][0] for distance in tempDist]
        digitsMatch = [rawData[ind][0] for ind in indexes]

        matches.append((label1, mode(digitsMatch)[0][0]))

    amtCorrect = 0
    for match in matches:
        amtCorrect += int(match[0] == match[1])

    accuracy = amtCorrect / len(matches)
    return matches, accuracy


def actualKNN(rawData, testData, k=3):
    testLabels = []
    for testValue in testData:
        distances = [np.linalg.norm(np.array(testValue) - np.array(digitData[1:])) for digitData in rawData]
        copyDistances = np.sort(np.array(distances))
        closestDistances = copyDistances[:k]

        indexes = [np.where(distances == closest)[0][0] for closest in closestDistances]
        digitsMatch = [rawData[ind][0] for ind in indexes]
        testLabels += [mode(digitsMatch)[0][0]]

    return testLabels


def crossValidation(rawData):
    kf = KFold(n_splits=3)
    trainingData = np.array(list(rawData))
    np.random.shuffle(trainingData)
    accuracy = 0
    allTrainLabels, allTestLabels = [], []

    for trainIndex, testIndex in kf.split(trainingData):
        trainValues = trainingData[trainIndex]
        testValues = [data[1:] for data in trainingData[testIndex]]
        testLbls = [data[0] for data in trainingData[testIndex]]

        # call KNN and get accuracy
        knnTestLabels = actualKNN(trainValues, testValues)

        tempAccuracy = 0
        for index, value in enumerate(knnTestLabels):
            tempAccuracy += int(value == testLbls[index])

        accuracy += tempAccuracy / len(knnTestLabels)
        allTrainLabels += testLbls  # actual "test" labels
        allTestLabels += knnTestLabels  # predicted "test" labels

    confusionMatrix = confusion_matrix(allTrainLabels, allTestLabels)

    return round(accuracy / 3.0, 2), confusionMatrix


if __name__ == '__main__':
    trainData = extractData("train.csv")

    # plotDigits(trainData)
    # digitFrequency(data)
    # # findClosestNeighbor(data[:3000])
    # print("Binary comparison in progress...")
    # genuine, impostor = binaryComparison(trainData)
    # print("Generating ROC...")
    # genROC(genuine, impostor)
    # matches, accuracy = knnClassifier(data, 7)

    # print(crossValidation(trainData))

    #
    # testData = extractData("test.csv")
    # actualKNN(trainData[:30001], testData)
