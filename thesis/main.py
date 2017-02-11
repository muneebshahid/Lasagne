#code adapted from minst.py in Lasagne
import numpy as np
import theano
from theano import tensor as T
import theano.printing as pr

import lasagne
import time


def load_mnist():
    import gzip

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    root_datatsets = '../../datasets/mnist/'

    X_train = load_mnist_images(root_datatsets + 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(root_datatsets + 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(root_datatsets + 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(root_datatsets + 't10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_cnn_mnist(input_var):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def main(args, num_epochs=500):
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    input_var = args['input_var']
    target_var = args['target_var']
    network = args['network']

    # training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)


    loss_prev = T.fscalar('loss_prev')
    if args['optim'] == 'eve':
        print 'using eve'
        updates = lasagne.updates.eve(loss, params, loss_prev)
    else:
        print 'using adam'
        updates = lasagne.updates.adam(loss, params)

    train_fn = theano.function([input_var, target_var, loss_prev], loss, updates=updates, on_unused_input='ignore')


    # testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    train_log_file_name  = args['optim'] + '.train'
    test_log_file_name  = args['optim'] + '.test'
    print('Starting Training...')    
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        batch_loss = 0
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            batch_loss = train_fn(inputs, targets, batch_loss)
            train_err += batch_loss
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        epoch_time_taken = time.time() - start_time
        epoch_train_loss = train_err / train_batches
        epoch_val_loss = val_err / val_batches
        epoch_val_acc = val_acc / val_batches * 100
        
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, epoch_time_taken))
        print("  training loss:\t\t{:.6f}".format(epoch_train_loss))
        print("  validation loss:\t\t{:.6f}".format(epoch_val_loss))
        print("  validation accuracy:\t\t{:.2f} %".format(epoch_val_acc))

        with open(train_log_file_name, 'a')  as train_file:
            train_file.write("{:.3f}".format(epoch_time_taken) + "\t{:.6f}".format(epoch_train_loss) + "\t{:.6f}".format(epoch_val_loss) + "\t{:.2f}".format(epoch_val_acc) + "\n")

        # Print and write test values
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        final_test_loss = test_err / test_batches
        final_test_acc = test_acc / test_batches * 100
        print("Test Results:")
        print("  test loss:\t\t\t{:.6f}".format(final_test_loss))
        print("  test accuracy:\t\t{:.2f} %".format(final_test_acc))
        with open(test_log_file_name, 'a') as test_file:
            test_file.write("{:.6f}".format(final_test_loss) + "\t{:.2f}".format(final_test_acc) + "\n")
        print("--------\n")

if __name__ == '__main__':
    mnist = True
    optim = 'adam'
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if mnist:
        network = build_cnn_mnist(input_var)
        dataset = load_mnist()

    args = {
            'network': network, \
            'dataset': dataset, \
            'input_var': input_var, \
            'target_var': target_var, \
            'optim': optim
            }
    main(args)
