from sklearn.datasets import fetch_openml
from numpy import dot, linalg, mean
from pandas import DataFrame , concat
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time


mnist = fetch_openml('mnist_784')
mnist_data: DataFrame = mnist.data[:10000]
mnist_target: DataFrame = mnist.target[:10000]

trainset = mnist_data[:10000 * 8//10]
testset = mnist_data[10000 * 8//10:]

kfold = 10

def metric(q,p):
    innerproduct = dot(q,p)
    magnitudeproduct = linalg.norm(q) * linalg.norm(p)
    return innerproduct / magnitudeproduct

def predict(x: DataFrame, trainset: DataFrame, k: int):
    neighbors = []
    
    for i in range(len(trainset.index)):
        tindex = trainset.index[i]
        similarity = metric(x.to_numpy(), trainset.iloc[i].to_numpy())
        
        neighbors = sorted(neighbors, key=lambda n: -n['similarity'])
        
        if len(neighbors) < k:
            neighbors.append({'lable': mnist_target.loc[tindex], 'similarity': similarity})
        elif similarity > neighbors[-1]['similarity']:
            neighbors.pop()
            neighbors.append({'lable': mnist_target.loc[tindex], 'similarity': similarity})
    counts = [0] * 10
    for n in neighbors:
        counts[int(n['lable'])] += 1
    maxcount = max(counts)
    lable = [i for i, count in enumerate(counts) if count == maxcount]
    
    return str(lable[0])


def kaccuracy(k: int):
    accuracy = []
    for i in range(kfold):
        firstindex = i *(len(trainset.index)//kfold)
        lastindex = (i+1)*(len(trainset.index)//kfold)
        
        kfoldset = trainset[firstindex: lastindex]
        newtrainset = concat([trainset[0: firstindex], trainset[lastindex:]])
        
        rightclassification = 0
        for i in range(len(kfoldset.index)):
            predictlable = predict(kfoldset.iloc[i], newtrainset, k)
            reallable = mnist_target[kfoldset.index[i]]
            if (predictlable == reallable):
                rightclassification += 1
        
        accuracy.append(float('%.2f' % (rightclassification / len(kfoldset.index))))
        
    return float('%.2f' % mean(accuracy))

# print (f"Accuracy for k=1: {kaccuracy(1)}")
# print (f"Accuracy for k=2: {kaccuracy(2)}")
# print (f"Accuracy for k=3: {kaccuracy(3)}")
# print (f"Accuracy for k=4: {kaccuracy(4)}")
# print (f"Accuracy for k=5: {kaccuracy(5)}")
# print (f"Accuracy for k=6: {kaccuracy(6)}")
# print (f"Accuracy for k=7: {kaccuracy(7)}")
# print (f"Accuracy for k=8: {kaccuracy(8)}")
# print (f"Accuracy for k=9: {kaccuracy(9)}")
# print (f"Accuracy for k=10: {kaccuracy(10)}")

# accuracies = []
# usedK = []
# for i in range(15):
#     accuracies.append(kaccuracy((2*i)+1))
#     usedK.append((2*i)+1)




accuracies = []
usedK = []

for i in range(15):
    k = (i*2) + 1

    usedK.append(k)

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(trainset, mnist_target.loc[trainset.index])
    predicts = knn.predict(trainset)

    rightClassification = 0
    for j in range(len(trainset.index)):
        predictLabel = predicts[j]
        realLabel = mnist_target[trainset.index[j]]
        if (predictLabel == realLabel):
            rightClassification += 1

    accuracies.append(
        (rightClassification / len(trainset.index)))
        
plt.figure(figsize=(10, 5))
plt.plot(usedK, accuracies, marker='o', linestyle='dashed', color='b')
plt.title('Cross validation scores for different k')
plt.xlabel('K')
plt.ylabel('Cross validation accuracy')
plt.grid(True)
plt.show()


