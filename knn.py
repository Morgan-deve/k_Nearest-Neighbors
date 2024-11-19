from sklearn.datasets import fetch_openml
from numpy import dot, linalg, mean
from pandas import DataFrame , concat


mnist = fetch_openml('mnist_784')
mnist_data: DataFrame = mnist.data[:1000]
mnist_target: DataFrame = mnist.target[:1000]

trainset = mnist_data[:1000 * 8//10]
testset = mnist_data[1000 * 8//10:]

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

print (f"Accuracy for k=1: {kaccuracy(1)}")
print (f"Accuracy for k=2: {kaccuracy(2)}")
print (f"Accuracy for k=3: {kaccuracy(3)}")
print (f"Accuracy for k=4: {kaccuracy(4)}")
print (f"Accuracy for k=5: {kaccuracy(5)}")
print (f"Accuracy for k=6: {kaccuracy(6)}")
print (f"Accuracy for k=7: {kaccuracy(7)}")
print (f"Accuracy for k=8: {kaccuracy(8)}")
print (f"Accuracy for k=9: {kaccuracy(9)}")
print (f"Accuracy for k=10: {kaccuracy(10)}")
        
