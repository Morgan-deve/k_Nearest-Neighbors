from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784')
mnist_data = mnist.data[:10000].to_numpy()
mnist_target = mnist.target[:10000].to_numpy()
