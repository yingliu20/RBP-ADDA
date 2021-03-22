import numpy as np
import tensorflow as tf


'''
Helper function to construct a convolutional layer given its parameres
'''



def create_cnn_layer(data, weights_matrix, bias_vector, strides_x_y):
    all_strides = [1, strides_x_y[0], strides_x_y[1], 1]
    result = tf.nn.conv2d(data, weights_matrix, strides=all_strides, padding='VALID')
    result = tf.nn.bias_add(result, bias_vector)
    result = tf.nn.relu(result)
    return result

'''
Helper function to construct a pooling layer given its parameres
'''
def create_pooling_layer(data, kpool_x_y, strides_x_y,avg=False):
    all_kpools = [1, kpool_x_y[0], kpool_x_y[1], 1]
    all_strides = [1, strides_x_y[0], strides_x_y[1], 1]
    result = tf.nn.max_pool(data, ksize=all_kpools, strides=all_strides, padding='VALID')
    result2 = tf.nn.avg_pool(data, ksize=all_kpools, strides=all_strides, padding='VALID')
    result = tf.concat([result,result2],axis=-1)
    return result

'''
Helper function to construct two layers of convolution and max pooling
'''
def create_layer(data, weights_matrix, bias_vector, strides_x_y, kpool_x_y,avg=False):
    result = create_cnn_layer(data, weights_matrix, bias_vector, strides_x_y)
    result = create_pooling_layer(result, kpool_x_y, strides_x_y,avg=avg)
    return result

'''
Helper function to construct a CNN (convolution and pooling) with multiple filters
'''
def create_multiple_filter_cnn(data, layer, filters, weights, biases, strides_x_y, kpool_x_y):
    pooled_outputs = []
    for filter_index in range(len(weights)):
        f = filters[filter_index]
        W = weights[filter_index]
        b = biases[filter_index]
        s = strides_x_y[filter_index]
        k = kpool_x_y[filter_index]
        filter_output = create_layer(data, W, b, s, k)
        print ("FILTER OUTPUT", tf.shape(filter_output), filter_output.get_shape())
        pooled_outputs.append(filter_output)
    cnn_output = tf.concat(pooled_outputs, 3)
    return cnn_output


'''
Helper function to flatten a CNN output
'''
def flatten(conv_data, fc_size):
    flat_data = tf.reshape(conv_data, [-1, fc_size])
    return flat_data

'''
Helper function to perform a linear transformation with possible non linear activation
'''
def nn_layer(data, weights, bias, activate_non_linearity=True,use_bias=True,bn=False,is_training=None):
    if use_bias:
        result = tf.add(tf.matmul(data, weights), bias)
    else:
        result = tf.matmul(data, weights)

    if bn:
        result = tf.layers.batch_normalization(result, momentum=0.95, training=is_training, renorm=False, fused=True)

    if activate_non_linearity:
        result = tf.nn.relu(result)

    return result

'''
Helper function to compute pearson correlation between to row vectos
'''
def pearson_correlation(x, y):
    mean_x, var_x = tf.nn.moments(x, [0])
    mean_y, var_y = tf.nn.moments(y, [0])
    std_x = tf.sqrt(var_x)
    std_y = tf.sqrt(var_y)
    mul_vec = tf.multiply((x - mean_x), (y - mean_y))
    covariance_x_y, _ = tf.nn.moments(mul_vec, [0])
    pearson = covariance_x_y / (std_x * std_y)
    return pearson

