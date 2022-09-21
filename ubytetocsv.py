#script to pull MNIST data and make into csv
#based off http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/#example-1-part-2-loading-mnist-into-numpy-arrays

from mlxtend.data import loadlocal_mnist
import platform

if not platform.system() == 'Windows':
    X, y = loadlocal_mnist(
            images_path='train-images-idx3-ubyte',
            labels_path='train-labels-idx1-ubyte')

else:
    X, y = loadlocal_mnist(
            images_path='train-images.idx3-ubyte',
            labels_path='train-labels.idx1-ubyte')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])
