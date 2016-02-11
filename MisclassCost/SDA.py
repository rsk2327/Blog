# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:23:17 2015

@author: rsk
"""

"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import os
import sys
import time
import cPickle
import gzip
import numpy


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from LogLayer import *
from MLP import *
from Autoencoder import *

from ClassExtractor import imbalance




# start-snippet-1
class SdA(object):
   
   

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        W = None,
        b = None,
        corruption_levels=[0.1, 0.1],
        misClassCost=[1,2,3]

    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        self.misclass = T.ivector('weight')
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2

        W_temp = []
        b_temp = []

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
                
            if W is not None:
                W_temp.append(W[i])
                b_temp.append(b[i])
                
            else:
                W_temp.append(None)
                b_temp.append(None)

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        W = W_temp[i],
                                        b = b_temp[i],
                                        activation=T.nnet.sigmoid)
                                        
#            print 'Hidden Layer created'
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            
            self.params.extend(sigmoid_layer.params)
            
#            print 'W: ', (sigmoid_layer.W).shape

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
            
#            print 'dA constructed'
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        if W is not None:
            W_temp.append(W[len(hidden_layers_sizes)])
            b_temp.append(W[len(hidden_layers_sizes)])
        else:
            W_temp.append(None)
            b_temp.append(None)
            
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs,
            W = W_temp[len(hidden_layers_sizes)],
                       b = b_temp[len(hidden_layers_sizes)]
        )
        
        

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y,self.misclass)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        
    #def predict(self,data,index,batch_size):
        
        
        
        #val = self.logLayer.predict()
       # 
      #  fn = theano.function(inputs=[index], outputs=val,givens = {self.x:testerdata[index*batch_size :(index+1)*batch_size] })

    def pretraining_functions(self, train_set_x, batch_size):
    

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate,sparsity_level = 0.001,sparsity_reg = 0.01)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.001)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):


        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        misclassLabels = datasets[3]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        print '@@@@@@@@@@@@@@@@'
        print self.params
        print '@@@@@@@@@@@@@@@@'
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.misclass : misclassLabels[index*(batch_size) : (index+1)*batch_size]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_SdA(misClassDict,finetune_lr=0.005, pretraining_epochs=35,
             pretrain_lr=0.001, training_epochs=50,
             dataset='mnist.pkl.gz', batch_size=10,layer_sizes=[1000,500,100],corruption_levels=[.1,.2,.3]):

    print "preparing the dataset"
    
    dataset = "/home/rsk/Documents/MNIST/Misclassification/mnist.pkl.gz"
    f = gzip.open(dataset, 'rb')
    [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)] = cPickle.load(f)
    f.close()

    
    #[(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)] = dataset
    
    [train_set_x,train_set_y] = imbalance(train_set_x,train_set_y,digits={3:0.5})
    #[valid_set_x,valid_set_y] = imbalance(valid_set_x,valid_set_y,digits={3:0.5})
    
    #defining the misclass cost weights
    labels=[]
    for i in range(len(train_set_y)):
        labels.append(misClassDict[train_set_y[i]])
    
    train_set_x = theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True),'int32')
    valid_set_x = theano.shared(numpy.asarray(valid_set_x,dtype=theano.config.floatX),borrow=True)
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y,dtype=theano.config.floatX),borrow=True),'int32')
    test_set_x = theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=True)
    test_set_y = T.cast(theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=True),'int32')
    labels = T.cast(theano.shared(numpy.asarray(labels,dtype=theano.config.floatX),borrow=True),'int32')
    
#    train_set_x = theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
#    train_set_y =theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True)
#    valid_set_x = theano.shared(numpy.asarray(valid_set_x,dtype=theano.config.floatX),borrow=True)
#    valid_set_y = theano.shared(numpy.asarray(valid_set_y,dtype=theano.config.floatX),borrow=True)
#    test_set_x = theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=True)
#    test_set_y = theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=True)
#    labels = theano.shared(numpy.asarray(labels,dtype=theano.config.floatX),borrow=True)
    
    testerdata = theano.shared(numpy.asarray(tests_data,dtype = theano.config.floatX),borrow = True)
    
    datasets = [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y),labels]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=layer_sizes,
        n_outs=10
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    index = T.iscalar()
    
    pred_model = theano.function([index],outputs = sda.logLayer.predict(),givens = {sda.x:testerdata[index*batch_size :(index+1)*batch_size]})
    
    
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = corruption_levels
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    print "making predictions"
    
    preds=[]
    for index in range(28000/batch_size):
        preds.append(pred_model(index))
                          
                          
    return [preds,batch_size]


if __name__ == '__main__':
    
    print '..loading test data'
    tests1_data=numpy.genfromtxt('/home/rsk/Documents/MNIST/test.csv',delimiter=',')
    tests_data = [i for i in tests1_data]
    
    #misClassDict = {0:1.0,1:1.0,2:1.0,3:3.0,4:1.0,5:1.0,6:1.0,7:1.0,8:1.0,9:1.0}
    misClassDict = {0:1,1:1,2:1,3:3,4:1,5:1,6:1,7:1,8:1,9:1}
    actual = test_SdA(misClassDict = misClassDict)
    
    f = open('/home/rsk/Documents/MNIST/SDAsol.csv','w')
    f.write('ImageId,label\n')
    
    index = 1
    for i in range(28000/actual[1]):
        for j in range(actual[1]):
            f.write(str(index)+','+str(actual[0][i][j])+'\n')
            index =  index + 1
    f.close()
