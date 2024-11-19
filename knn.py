from sklearn.datasets import fetch_openml
from numpy import dot, linalg
from pandas import DataFrame


mnist = fetch_openml('mnist_784')
mnist_data: DataFrame = mnist.data[:10000]
mnist_target: DataFrame = mnist.target[:10000]

trainset = mnist_data[:10000 * 8//10]
testset = mnist_data[10000 * 8//10:]


def metric(q,p):
    innerproduct = dot(q,p)
    magnitudeproduct = linalg.norm(q) * linalg.norm(p)
    return innerproduct / magnitudeproduct

def predict(x, k: int, trainset):
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


