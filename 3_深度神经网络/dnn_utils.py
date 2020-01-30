import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    
    return A

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    return A


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def load_data():   
    #加载训练数据
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5',"r")
    #从训练数据中提取出图片的特征数据
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    #从训练数据中提取出图片的标签数据
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])
    
    #加载测试数据
    test_dataset = h5py.File('./datasets/test_catvnoncat.h5',"r")
    #从测试数据中提取出图片的特征数据
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    #从测试数据中提取出图片的标签数据
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])
    
    #加载标签类别数据，这里类别有两种，1代表有猫，0代表无猫
    #标签数据中只是0，1   这里是0，1的解释array([b'non-cat', b'cat'], dtype='|S7')
    classes = np.array(test_dataset['list_classes'][:])
    
    #train_set_y_orig 原shape 为(209,) 一维数组
    #reshape后 为（1，209） 二维数组
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    
    #test_set_y_orig 原shape 为(50,)
    #reshape后 为 （1，50）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])) # 从(50,)变成(1, 50)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes