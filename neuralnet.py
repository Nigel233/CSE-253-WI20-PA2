################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import copy


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    imax, imin = img.max(axis=1)[:, None], img.min(axis=1)[:, None]
    img = (img-imin) / (imax-imin)
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    ohe = np.zeros((np.shape(labels)[0], num_classes))
    for idx, l in enumerate(labels):
        ohe[idx][l] = 1
    return ohe


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    exp = x - np.max(x, axis=1)[:, None]
    exp = np.exp(exp)
    return exp / np.sum(exp, axis=1)[:, None]


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        self.x = x
        return 1/(1+np.exp(-x))
        # raise NotImplementedError("Sigmoid not implemented")

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x = x
        return np.tanh(x)
        # raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = x
        return np.maximum(0,x)
        # raise NotImplementedError("ReLu not implemented")

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        y = self.sigmoid(self.x)
        return y*(1-y)
        # raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        y = self.tanh(self.x)
        return 1-np.power(y,2)
        # raise NotImplementedError("tanh gradient not implemented")

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return (self.x > 0).astype('float')
        # raise NotImplementedError("ReLU gradient not implemented")


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.normal(0, 1, (in_units, out_units))    # Declare the Weight matrix
        self.b = np.zeros(out_units)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.v_w = np.zeros((in_units, out_units))
        self.v_b = np.zeros(out_units)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = np.dot(self.x, self.w) + self.b
        return self.a
        # raise NotImplementedError("Layer forward pass not implemented.")

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_w = np.dot(self.x.T, delta) / np.shape(delta)[0]
        self.d_b = np.mean(delta, axis=0)
        self.d_x = np.sum(np.dot(self.w, delta.T), axis=1) / np.shape(delta)[0]
        return self.d_x
        # raise NotImplementedError("Backprop for Layer not implemented.")



class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        y = self.x
        for lyr in self.layers:
            y = lyr(y)
        self.y = softmax(y)

        loss = None
        if not targets is None:
            self.targets = targets
            loss = self.loss(self.y, self.targets)
        return self.y, loss
        # raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        ll = -np.sum(np.log(logits)*targets)
        return ll/len(logits)
        # raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        dlt = self.targets - self.y
        for lyr in self.layers[::-1]:
            dlt = lyr.backward(dlt)
        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")

    def renew(self, lr, L2, momentum, gamma):
        for lyr in self.layers:
            if isinstance(lyr, Layer):
                if momentum:
                    lyr.v_w = gamma*lyr.v_w + (1-gamma)*lyr.d_w
                    lyr.v_b = gamma*lyr.v_b + (1-gamma)*lyr.d_b
                    lyr.b += lr*lyr.v_b
                    lyr.w += lr*(lyr.v_w - L2*lyr.w)
                else:
                    lyr.b += lr*lyr.d_b
                    lyr.w += lr*(lyr.d_w - L2*lyr.w)

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    batch_size = config['batch_size']
    nbatchs = int(len(x_train)/batch_size)

    last_loss = 0
    incre_epo = 0

    epol = []
    trainll, trainal = [], []
    valll, valal = [], []

    for epo in range(config['epochs']):
        for itr in range(nbatchs):
            x = x_train[itr*batch_size: (itr+1)*batch_size]
            y = y_train[itr*batch_size: (itr+1)*batch_size]
            model(x, y)
            model.backward()
            model.renew(config['learning_rate'], config['L2_penalty'], config['momentum'], config['momentum_gamma'])
    
        train_acc, train_loss = test(model, x_train, y_train)
        val_acc, val_loss = test(model, x_valid, y_valid)
        if epo % 10 == 0:
            print("epo: {} - tr_loss: {} - tr_acc: {} - val_loss: {} - val_acc: {}".format(epo, train_loss, train_acc, val_loss, val_acc))
            # trainll.append(train_loss)
            # trainal.append(train_acc)
            # valll.append(val_loss)
            # valal.append(val_acc)
            # epol.append(epo)

        if config['early_stop']:
            if val_loss > last_loss:
                incre_epo += 1
                if incre_epo == config['early_stop_epoch']:
                    # print("Early Stopped.")
                    # print("tl = ", trainll)
                    # print("ta = ", trainal)
                    # print("vl = ", valll)
                    # print("va = ", valal)
                    # print("l_x = ", epol)
                    return
            else:
                incre_epo = 0
        last_loss = val_loss
    # print("tl = ", trainll)
    # print("ta = ", trainal)
    # print("vl = ", valll)
    # print("va = ", valal)
    # print("l_x = ", epol)
    return
    # raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    y_pred, loss = model(X_test, y_test)
    acc = np.sum(np.argmax(y_pred, axis=1)==np.argmax(y_test, axis=1)) / len(y_test)
    return acc, loss
    # raise NotImplementedError("Test method not implemented")

def approx(model, datax, datay):
    sx, sy = [], []
    label = 0
    print(np.shape(datax), np.shape(datay))
    for i in range(len(datax)):
        if np.argmax(datay[i]) == label:
            sx.append(datax[i:i+1])
            sy.append(datay[i:i+1])
            label+=1
        if label == 10:
            break

    ori = copy.deepcopy(model);
    print("==========")
    for i in range(10):
        model = copy.deepcopy(ori)
        model.layers[2].b[0] += 10e-2
        _, eplus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model.layers[2].b[0] -= 10e-2
        _, eminus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model(sx[i], sy[i])
        model.backward()
        de = model.layers[2].d_b[0]

        print("{:.5e} \t {:.5e} \t {:.5e} \t {:.5e} \t {:.5e}".format(eplus,eminus, (eplus-eminus)/(2*10e-2),-de, (eplus-eminus)/(2*10e-2)+de))
        
    print("==========")
    for i in range(10):
        model = copy.deepcopy(ori)
        model.layers[0].b[0] += 10e-2
        _, eplus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model.layers[0].b[0] -= 10e-2
        _, eminus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model(sx[i], sy[i])
        model.backward()
        de = model.layers[0].d_b[0]
    
        print("{:.5e} \t {:.5e} \t {:.5e} \t {:.5e} \t {:.5e}".format(eplus,eminus, (eplus-eminus)/(2*10e-2),-de, (eplus-eminus)/(2*10e-2)+de))
        
    print("==========")
    # hidden to output weight 1
    for i in range(10):
        # output bias
        model = copy.deepcopy(ori)
        model.layers[2].w[0][0] += 10e-2
        _, eplus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model.layers[2].w[0][0] -= 10e-2
        _, eminus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model(sx[i], sy[i])
        model.backward()
        de = model.layers[2].d_w[0][0]

        print("{:.5e} \t {:.5e} \t {:.5e} \t {:.5e} \t {:.5e}".format(eplus,eminus, (eplus-eminus)/(2*10e-2),-de, (eplus-eminus)/(2*10e-2)+de))
        
    print("==========")
    # hidden to output wight 2
    for i in range(10):
        # output bias
        model = copy.deepcopy(ori)
        model.layers[2].w[1][1] += 10e-2
        _, eplus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model.layers[2].w[1][1] -= 10e-2
        _, eminus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model(sx[i], sy[i])
        model.backward()
        de = model.layers[2].d_w[1][1]

        print("{:.5e} \t {:.5e} \t {:.5e} \t {:.5e} \t {:.5e}".format(eplus,eminus, (eplus-eminus)/(2*10e-2),-de, (eplus-eminus)/(2*10e-2)+de))
        
    print("==========")
    # input to hidden weight 1
    for i in range(10):
        # output bias
        model = copy.deepcopy(ori)
        model.layers[0].w[300][30] += 10e-2
        _, eplus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model.layers[0].w[300][30] -= 10e-2
        _, eminus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model(sx[i], sy[i])
        model.backward()
        de = model.layers[0].d_w[300][30]

        print("{:.5e} \t {:.5e} \t {:.5e} \t {:.5e} \t {:.5e}".format(eplus,eminus, (eplus-eminus)/(2*10e-2),-de, (eplus-eminus)/(2*10e-2)+de))
    print("==========")
    # input to output weight 2
    for i in range(10):
        model = copy.deepcopy(ori)
        model.layers[0].w[400][19] += 10e-2
        _, eplus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model.layers[0].w[400][19] -= 10e-2
        _, eminus = model(sx[i], sy[i])

        model = copy.deepcopy(ori)
        model(sx[i], sy[i])
        model.backward()
        de = model.layers[0].d_w[400][19]

        print("{:.5e} \t {:.5e} \t {:.5e} \t {:.5e} \t {:.5e}".format(eplus,eminus, (eplus-eminus)/(2*10e-2),-de, (eplus-eminus)/(2*10e-2)+de))
        
        


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # x_valid, y_valid = ...
    div = int(len(x_train)*0.8)
    x_val, y_val = x_train[div:], y_train[div:]
    x_train, y_train = x_train[:div], y_train[:div]

    # train the model
    train(model, x_train, y_train, x_val, y_val, config)

    # approx(model, x_test, y_test)

    test_acc, loss = test(model, x_test, y_test)
    print("test acc: {}".format(test_acc))
