import numpy as np

def getTitles():
    titles = []
    with open("science2k-titles.txt") as tFile:
        for title in tFile.readlines():
            titles += [title.rstrip()]   
    return titles

def getVocab():
    vocab = []
    with open("science2k-vocab.txt") as tFile:
        for v in tFile.readlines():
            vocab += [v.rstrip()]
    return vocab

def kMeans():
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    docWordData = np.load("science2k-doc-word.npy")

    numClusters = list(range(1,21))
    sse = [0] * len(numClusters)

    for k in numClusters:
        print(k)
        kmeans = KMeans(k, n_init=10, init='random').fit(docWordData)
        sse[k-1] += kmeans.inertia_
    
    plt.plot(list(range(1,21)), sse, marker='x')
    plt.title('SSE for different Ks')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.grid()
    plt.savefig("SSE for Ks")

# 2.a
def getTop(filename, getFn):
    from sklearn.cluster import KMeans
    from collections import defaultdict
    docWordData = np.load(filename)

    # K was chosen to be 10 based on graph generated from kMeans()
    kmeans = KMeans(10, n_init=10, init='random').fit(docWordData)
    labels = kmeans.labels_

    # cluster: [index of documents in cluster]
    clusters = defaultdict(list)
    for index, cluster in enumerate(labels):
        clusters[cluster] += [index]
    
    centers = kmeans.cluster_centers_
    distances = defaultdict(list)
    for index, center in enumerate(centers):
        # for every document caluclate mx - xbar
        for doc in clusters[index]:
            distance = np.sum(np.subtract(center, docWordData[doc]))
            distance = np.sqrt(np.square(distance))
            distances[index].append((doc, distance))
    
    docInfo = getFn()
    topDocs = [[] for _ in range(len(clusters.keys()))]

    # get closest 10 documents to the center of each cluster
    for i, dists in distances.items():
        distances[i] = sorted(dists, key= lambda d: d[1])[:10]
        for d in distances[i]:
            topDocs[i] += [docInfo[d[0]]]

    return topDocs

def write2File(filename, top):
    with open(filename, "w") as dFile:
        
        for k, doc in enumerate(top):
            print("For cluster {}, {} items were written".format(k, len(doc)))
            dFile.write("k = {}:\n".format(k))
            dFile.write(', '.join(doc) + '\n')


if __name__ == "__main__":
    # print("Document clustering... \n")
    # docWordFile = "science2k-doc-word.npy"
    # docs = getTop(docWordFile, getTitles)
    # filename = "topDocs.txt"
    # write2File(filename, docs)

    print("Vocab clustering...\n")
    docWordFile = "science2k-word-doc.npy"
    docs = getTop(docWordFile, getVocab)
    filename = "topVocab.txt"
    write2File(filename, docs)

