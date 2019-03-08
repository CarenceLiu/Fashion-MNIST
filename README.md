# Fashion MNIST

## Environments

If your package version is not the same as listed below, there is a chance that you may run the code successfully. But still, please use the recommended environment setting.

| Environment | Version |
| ----------- | ------- |
| python      | 3.6.6   |
| tensorflow  | 1.10.0  |
| numpy       | 1.16.0  |
| mnist       | --      |

The ``mnist`` package is a mnist data parser, which can be installed with ``pip install python-mnist``. It can be imported with ``import mnist`` in your own code.

## The Fashion MNIST Dataset

The Fashion MNIST dataset is from [here](https://www.kaggle.com/zalando-research/fashionmnist).

It is a dataset a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Each training and test example is assigned to one of the following labels: t-shirt/top (0), trouser (1), pullover (2), dress (3), coat (4), sandal (5), shirt (6), sneaker (7), bag (8), ankle boot (9).

The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."
