#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: training_policy.py
# Date: Thu Sep 18 10:23:59 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

import numpy as np
from lib.progress import Progressor
import sys

class TrainPolicy(object):

    def __init__(self, train_model, valid_model, 
                 n_batches, logger, learning_rate_provider, test_model = None):
        self.train_model = train_model
        self.test_model = test_model
        self.valid_model = valid_model
        self.n_batches = n_batches
        self.valid_freq = n_batches[0] / 2
        self.test_freq = n_batches[0] * 2
        self.logger = logger
        self.learning_rate_provider = learning_rate_provider

    def work(self):
        pass
    def flush(self):
        sys.stdout.flush()

    def do_valid(self):
        valid_loss = [self.valid_model(i) for i
                     in xrange(self.n_batches[1])]
        return np.mean(valid_loss)

    def do_test(self):
        assert (self.test_model is not None)
        test_loss = [self.test_model(i) for i
                     in xrange(self.n_batches[2])]
        return np.mean(test_loss)



class TrainForever(TrainPolicy):

    def __init__(self, train_model, valid_model,
                 n_batches, logger, learning_rate_provider, test_model = None):
        super(TrainForever, self).__init__(train_model, valid_model,
                                           n_batches, logger, learning_rate_provider,test_model)

    def work(self):
        print 'Start training...'
        best_loss = np.inf
        epoch = 0
        progressor = Progressor(None)

        while True:
            epoch = epoch + 1
            if epoch > 1:
                progressor.report(1, True)
                # save params at the beginning of each epoch
                self.logger.save_params(epoch)

            learning_rate = self.learning_rate_provider.get_rate(epoch)
            print "In epoch {0}: learning rate is {1}".format(epoch, learning_rate)
            self.flush()
            for minibatch_index in xrange(self.n_batches[0]):
                iter = (epoch - 1) * self.n_batches[0] + minibatch_index

                if iter % 200 == 0 or (iter % 10 == 0 and iter < 30) or (iter < 5):
                    print 'training @ iter = ', iter
                    self.flush()
                cost_ij = self.train_model(minibatch_index, learning_rate)


                if (iter + 1) % self.valid_freq == 0 or iter in [5, 10, 20, 30, 60, 100, 200]:
                    # do a validation:

                    # compute zero-one loss on validation set
                    now_loss = self.do_valid()
                    print('After epoch %i, minibatch %i/%i, valid error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_batches[0], \
                           now_loss * 100.))

                    # got the best score until now
                    if now_loss < best_loss:
                        best_loss = now_loss
                        if best_loss < 0.85:
                            # save best params
                            print 'Yay! Saving best model ...'
                            self.logger.save_params('best')
                            if ( (iter+1) % self.test_freq ==0) and (self.test_model is not None):
                                test_score = self.do_test()
                                print('After epoch %i, minibatch %i/%i, with the best model, test error %f %%' % \
                                      (epoch, minibatch_index + 1, self.n_batches[0], \
                                       test_score * 100.))

                    else:
                        print('Fuck, now loss>best loss, best loss is %f %%, ' % \
                              (best_loss*100.) )
                    self.flush()

class TrainEarlyStopping(TrainPolicy):

    def __init__(self, train_model, valid_model,
                 n_batches, logger, learning_rate_provider,patience = 100000,patience_increase = 2,improvement_threshold = 0.995,
                 test_model = None):
        super(TrainEarlyStopping, self).__init__(train_model, valid_model,
                                                 n_batches, logger, learning_rate_provider,test_model)
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        

    def work(self):
        print 'Start training...'
        best_loss = np.inf
        epoch = 0
        progressor = Progressor(None)

        self.valid_freq = min(self.n_batches[0]/2, self.patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

        best_iter = 0
        done_looping = False

        while not done_looping:
            epoch = epoch + 1
            if epoch > 1:
                progressor.report(1, True)
                # save params at the beginning of each epoch
                self.logger.save_params(epoch)

            learning_rate = self.learning_rate_provider.get_rate(epoch)
            print "In epoch {0}: learning rate is {1}".format(epoch, learning_rate)
            self.flush()
            for minibatch_index in xrange(self.n_batches[0]):
                iter = (epoch - 1) * self.n_batches[0] + minibatch_index

                if iter % 200 == 0 or (iter % 10 == 0 and iter < 30) or (iter < 5):
                    print 'training @ iter = ', iter
                    self.flush()
                cost_ij = self.train_model(minibatch_index, learning_rate)


                if (iter + 1) % self.valid_freq == 0 or iter in [5, 10, 20, 30, 60, 100, 200]:
                    # do a validation:
                    
                    # compute zero-one loss on validation set
                    now_loss = self.do_valid()
                    print('After epoch %i, minibatch %i/%i, valid error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_batches[0], \
                           now_loss * 100.))

                    # got the best score until now
                    if now_loss < best_loss:

                        #improve patience if loss improvement is good enough
                        if now_loss < best_loss * self.improvement_threshold:
                            self.patience = max(self.patience, iter * self.patience_increase)
                        best_loss = now_loss
                        best_iter = iter
                        if best_loss < 0.85:
                            # save best params
                            print 'Yay! Saving best model ...'
                            self.logger.save_params('best')
                            if ( (iter+1) % self.test_freq ==0) and (self.test_model is not None):
                                test_score = self.do_test()
                                print('After epoch %i, minibatch %i/%i, with the best model, test error %f %%' % \
                                      (epoch, minibatch_index + 1, self.n_batches[0], \
                                       test_score * 100.))

                    else:
                        print('Fuck, now loss>best loss, best loss is %f %%, patience now is %i' %\
                              (best_loss*100. ,self.patience) )
                    self.flush()
                if self.patience <= iter:
                    done_looping = True
                    break

