import matplotlib.pyplot as plt  # For plotting
import numpy as np  # NumPy, for working with arrays/tensors
import os  # Built-in library for filesystem access etc.
import pickle  # For (re)storing Python objects into (from) files
import time  # For measuring time
import random  # Python's random library

# %matplotlib inline
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower


# Convolution feedforward
def conv_forward_naive(x, w, b, stride, padding):
    """
    The input consists of N samples, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height FH and width FW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, FH, FW)
    - b: Biases, of shape (F,)
    - stride: The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    - padding: The number of pixels that will be used to zero-pad the input.

    During padding, 'padding'-many zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * padding - FH) / stride
        W' = 1 + (W + 2 * padding - FW) / stride
    - cache: (x_pad, w, b, stride, padding)
    """

    # Layer output:
    out = None

    # Get values for the dimensions:
    N, C, H, W = x.shape
    F, _, FH, FW = w.shape

    # Just to make sure
    if (H - FH + 2 * padding) % stride != 0:
        return ValueError("Filters do not align horizontally")
    if (W - FW + 2 * padding) % stride != 0:
        return ValueError("Filters do not align vertically")

    ###########################################################################
    # TODO: Implement the convolutional forward pass. You cannot use          #
    # vector or matrix multiplications here. Be careful about stride.         #
    # Hints:                                                                  #
    #  (1) You can use the function np.pad for padding.                       #
    #  (2) You will have such a nested loop structure here:                   #
    #    - loop over N samples                                                #
    #      - loop over F filters                                              #
    #        - loop over rows of x_pad                                        #
    #          - loop over cols of x_pad                                      #
    #            - loop over kernel rows                                      #
    #              - loop over kernel cols                                    #
    #                - loop over channels                                     #
    ###########################################################################

    # Calculate output dimensions
    H_out = 1 + (H + 2 * padding - FH) // stride
    W_out = 1 + (W + 2 * padding - FW) // stride

    # Initialize output tensor
    out = np.zeros((N, F, H_out, W_out))

    # Pad input
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Perform convolution
    for n in range(N):
        for f in range(F):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + FH
                    w_start = w_out * stride
                    w_end = w_start + FW
                    out[n, f, h_out,
                        w_out] = np.sum(x_pad[n, :, h_start:h_end, w_start:w_end] * w[f, :, :, :]) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x_pad, w, b, stride, padding)
    return out, cache


# Convolution gradient
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, stride, padding) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x.
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x_pad, w, b, stride, padding = cache
    N, F, H_out, W_out = dout.shape
    _, C, H, W = x_pad.shape
    _, _, FH, FW = w.shape

    # Initialize gradients
    dx = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    # Compute gradients
    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f])
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + FH
                    w_start = w_out * stride
                    w_end = w_start + FW
                    dx[n, :, h_start:h_end, w_start:w_end] += dout[n, f, h_out, w_out] * w[f, :, :, :]
                    dw[f, :, :, :] += dout[n, f, h_out, w_out] * x_pad[n, :, h_start:h_end, w_start:w_end]

    # Remove padding from dx
    dx = dx[:, :, padding:-padding, padding:-padding]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dw, db


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


# Max pooling feedforward
def max_pool_forward_naive(x, stride, PH, PW):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - stride: The distance between adjacent pooling regions
    - PH: The height of each pooling region
    - PW: The width of each pooling region

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    H_out = 1 + (H - PH) // stride
    W_out = 1 + (W - PW) // stride

    out = np.zeros((N, C, H_out, W_out))

    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + PH
                    w_start = w_out * stride
                    w_end = w_start + PW
                    out[n, c, h_out, w_out] = np.max(x[n, c, h_start:h_end, w_start:w_end])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x, stride, PH, PW)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, stride, PH, PW) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    x, stride, PH, PW = cache
    N, C, H, W = x.shape
    H_out, W_out = dout.shape[2], dout.shape[3]

    dx = np.zeros_like(x)

    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + PH
                    w_start = w_out * stride
                    w_end = w_start + PW
                    x_pooling_region = x[n, c, h_start:h_end, w_start:w_end]
                    max_val = np.max(x_pooling_region)
                    dx[n, c, h_start:h_end,
                       w_start:w_end] += (x_pooling_region == max_val) * dout[n, c, h_out, w_out]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx


if __name__ == "__main__":
    ### Convolution feedforward check
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    stride = 2
    padding = 1
    out, _ = conv_forward_naive(x, w, b, stride, padding)
    correct_out = np.array([[[[-0.08759809, -0.10987781], [-0.18387192, -0.2109216]],
                             [[0.21027089, 0.21661097], [0.22847626, 0.23004637]],
                             [[0.50813986, 0.54309974], [0.64082444, 0.67101435]]],
                            [[[-0.98053589, -1.03143541], [-1.19128892, -1.24695841]],
                             [[0.69108355, 0.66880383], [0.59480972, 0.56776003]],
                             [[2.36270298, 2.36904306], [2.38090835, 2.38247847]]]])

    def rel_error(x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    # Compare your output to ours; difference should be around e-8
    print('Testing conv_forward_naive')
    print('difference: ', rel_error(out, correct_out))

    ### Convolution gradient check
    np.random.seed(231)
    x = np.random.randn(4, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2,)
    dout = np.random.randn(4, 2, 5, 5)
    stride = 1
    pad = 1

    dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, stride, pad)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, stride, pad)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, stride, pad)[0], b, dout)

    out, cache = conv_forward_naive(x, w, b, stride, pad)
    dx, dw, db = conv_backward_naive(dout, cache)

    # Your errors should be around e-8 or less.
    print('Testing conv_backward_naive function')
    print('dx error: ', rel_error(dx, dx_num))
    print('dw error: ', rel_error(dw, dw_num))
    print('db error: ', rel_error(db, db_num))

    ### Pooling feedforward check
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
    stride, PH, PW = (2, 2, 2)

    out, _ = max_pool_forward_naive(x, stride, PH, PW)

    correct_out = np.array([[[[-0.26315789, -0.24842105], [-0.20421053, -0.18947368]],
                             [[-0.14526316, -0.13052632], [-0.08631579, -0.07157895]],
                             [[-0.02736842, -0.01263158], [0.03157895, 0.04631579]]],
                            [[[0.09052632, 0.10526316], [0.14947368, 0.16421053]],
                             [[0.20842105, 0.22315789], [0.26736842, 0.28210526]],
                             [[0.32631579, 0.34105263], [0.38526316, 0.4]]]])

    # Compare your output with ours. Difference should be on the order of e-8.
    print('Testing max_pool_forward_naive function:')
    print('difference: ', rel_error(out, correct_out))

    ### Pooling gradient check
    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 4, 4)
    stride, PH, PW = (2, 2, 2)

    dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, stride, PH, PW)[0], x, dout)

    out, cache = max_pool_forward_naive(x, stride, PH, PW)
    dx = max_pool_backward_naive(dout, cache)

    # Your error should be on the order of e-12
    print('Testing max_pool_backward_naive function:')
    print('dx error: ', rel_error(dx, dx_num))
