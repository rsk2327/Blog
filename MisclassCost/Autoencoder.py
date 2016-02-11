# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:36:57 2015

@author: rsk
"""
import cPickle
import gzip
import os
import sys
import time
import numpy
from theano import *
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from LogLayer import *
from MLP import *


class dA(object):
    
    def __init__(self,numpy_rng,theano_rng=None,input=None,n_visible=784,n_hidden=500,W=None,bhid=None,bvis=None):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(low=-4*numpy.sqrt(6./(n_hidden+n_visible)), high =4*numpy.sqrt(6./(n_hidden+n_visible)),size=(n_visible,n_hidden) ),dtype = theano.config.floatX)
            W = theano.shared(value = initial_W,name='W',borrow=True)
        
        if not bvis:
            bvis = theano.shared( value=numpy.zeros(n_visible,dtype=theano.config.floatX),borrow=True)
            
        if not bhid:
            bhid = theano.shared( value=numpy.zeros(n_hidden,dtype=theano.config.floatX),borrow=True)
            
        
        self.W =W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
            
        
        self.params = [self.W,self.b,self.b_prime]
        
    
    def get_corrupted_input(self,input,corruption_level):
        
        return self.theano_rng.binomial(size=input.shape,n=1,p=1-corruption_level,dtype = theano.config.floatX)*input
        
    def get_hidden_values(self,input):
        return T.nnet.sigmoid(T.dot(input,self.W)+self.b)
        
    def get_reconstructed_input(self,hidden):
        
        return T.dot(hidden,self.W_prime)+ self.b_prime
        
    def kl_divergence(self, p, p_hat):
        term1 = p * T.log(p/p_hat)
        #term2 = p * T.log(p_hat)
        term3 = (1-p) * T.log((1 - p)/(1-p_hat))
       # term4 = (1-p) * T.log(1 - p_hat)
        return term1  + term3 
        
    def sparsity_penalty(self, y, sparsity_level=0.005, sparse_reg=1e-3):
#        if batch_size == -1 or batch_size == 0:
#            raise Exception("Invalid batch_size!")
        sparsity_level = T.extra_ops.repeat(sparsity_level, self.n_hidden)
        sparsity_penalty = 0
        avg_act = T.mean(y)
        kl_div = self.kl_divergence( sparsity_level,avg_act)
        sparsity_penalty = sparse_reg * kl_div.sum()
        # Implement KL divergence here.
        return sparsity_penalty
        
    def get_cost_updates(self,corruption_level,learning_rate,sparsity_level=0.005,sparsity_reg=0.001):
        
        
        tilde_x = self.get_corrupted_input(self.x,corruption_level)
        y = self.get_hidden_values(tilde_x)
        z =self.get_reconstructed_input(y)
        
        
        #the cross-entropy cost 
        #L = -T.sum(self.x*T.log(z) + (1-self.x)*T.log(1-z),axis = 1)
        sparsePenalty = self.sparsity_penalty(y,sparsity_level,sparsity_reg)
        cost = T.mean(((self.x-z)**2).sum(axis=1)) + sparsePenalty
        
        # the gradients of the weights
        gparams = T.grad(cost,self.params)
        
        updates = [(param,param - learning_rate*gparam) for param,gparam in zip(self.params,gparams)]
        
        return (cost,updates)
        
        
        

        
        
def run_code(learning_rate = 0.001,training_epochs = 60,dataset = 'mnist.pkl.gz',batch_size = 100 ):
    
    datasets = load_data(dataset)
    train_set_x,train_set_y = datasets[0]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    
    index = T.lscalar()
    x = T.matrix('x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))

    da  = dA(numpy_rng = rng,theano_rng = theano_rng,input = x,n_visible = 784,n_hidden = 500)

    cost,updates = da.get_cost_updates(corruption_level=0.3,learning_rate = learning_rate)

    train_da = theano.function([index],cost,updates = updates, givens = {x:train_set_x[index*batch_size : (index+1)*batch_size]})


    start_time = time.clock()

## training 

    for epoch in xrange(training_epochs):
        c=[]
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        
        print 'training epoch %d, cost ' % epoch, numpy.mean(c)
        
    end_time = time.clock()
    

    training_time = (end_time- start_time)
    
    print 'ran for %.2f m' % (training_time/60.)
    
    
        
        
if __name__ == '__main__':
    run_code()

        
        
        
            
        
        