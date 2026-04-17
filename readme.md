# Multilayer Perceptron (MLP) for Image Classification

An example implementation of a Multilayer Perceptron in PHP 8.3, created for educational purposes. 

To run the scripts, you need to install the `php-gd` library. It is used to parse PNG files.

```shell
sudo apt-get install php-gd
```

To train the model from scratch, delete the saved `weights` from the `model` folder and run the following script:

```shell
php train.php
```

The repository includes a pre-trained model. To verify its performance, run the command:

```shell
php test.php
```

The model is trained using the MNIST dataset. All external resources can be found in the References section.

## References

- [Mastering the Multi-Layer Perceptron (MLP) for Image Classification](https://medium.com/eincode/mastering-the-multi-layer-perceptron-mlp-for-image-classification-a0272baf1e29)
- [Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Neural Network Course in JavaScript](https://github.com/Jerga99/neural-network-course)
- [The MNIST database of handwritten digits](https://web.archive.org/web/20180630224356/http://yann.lecun.com/exdb/mnist/)
- [Simple script to convert MNIST to PNG format](https://github.com/myleott/mnist_png)
- [He/Xavier initialization & activation functions: choose wisely](https://github.com/christianversloot/machine-learning-articles/blob/main/he-xavier-initialization-activation-functions-choose-wisely.md)
