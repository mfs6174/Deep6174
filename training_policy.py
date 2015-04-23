#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: training_policy.py
# Date: Thu Sep 18 10:23:59 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

import numpy as np
from lib.progress import Progressor

class TrainPolicy(object):

    def __init__(self, train_model, test_model,
                 n_batches, logger, learning_rate_provider):
        self.train_model = train_model
        self.test_model = test_model
        self.n_batches = n_batches
        self.test_freq = n_batches[0] / 2
        self.logger = logger
        self.learning_rate_provider = learning_rate_provider

    def work(self):
        pass

class TrainForever(TrainPolicy):

    def __init__(self, train_model, test_model,
                 n_batches, logger, learning_rate_provider):
        super(TrainForever, self).__init__(train_model, test_model,
                 n_batches, logger, learning_rate_provider)

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
            for minibatch_index in xrange(self.n_batches[0]):
                iter = (epoch - 1) * self.n_batches[0] + minibatch_index

                if iter % 200 == 0 or (iter % 10 == 0 and iter < 30) or (iter < 5):
                    print 'training @ iter = ', iter
                cost_ij = self.train_model(minibatch_index, learning_rate)


                if (iter + 1) % self.test_freq == 0 or iter in [5, 10, 20, 30, 60, 100, 200]:
                    # do a validation:

                    # compute zero-one loss on validation set
                    test_loss = [self.test_model(i) for i
                                         in xrange(self.n_batches[2])]
                    now_loss = np.mean(test_loss)
                    print('After epoch %i, minibatch %i/%i, test error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_batches[0], \
                           now_loss * 100.))

                    # got the best score until now
                    if now_loss < best_loss:
                        best_loss = now_loss
                        if best_loss < 0.85:
                            # save best params
                            print 'Yay! Saving best model ...'
                            self.logger.save_params('best')
                    else:
                        print('Fuck, now loss>best loss, best loss is %f %%, ' % \
                              (best_loss*100.) )


class TrainEarlyStopping(TrainPolicy):

    def __init__(self, train_model, test_model,
                 n_batches, logger, learning_rate_provider,patience = 5000,patience_increase = 2,improvement_threshold = 0.995):
        super(TrainEarlyStopping, self).__init__(train_model, test_model,
                 n_batches, logger, learning_rate_provider)
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        

    def work(self):
        print 'Start training...'
        best_loss = np.inf
        epoch = 0
        progressor = Progressor(None)

        self.test_freq = min(self.n_batches[0]/2, self.patience / 2)
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
            for minibatch_index in xrange(self.n_batches[0]):
                iter = (epoch - 1) * self.n_batches[0] + minibatch_index

                if iter % 200 == 0 or (iter % 10 == 0 and iter < 30) or (iter < 5):
                    print 'training @ iter = ', iter
                cost_ij = self.train_model(minibatch_index, learning_rate)


                if (iter + 1) % self.test_freq == 0 or iter in [5, 10, 20, 30, 60, 100, 200]:
                    # do a validation:

                    # compute zero-one loss on validation set
                    test_loss = [self.test_model(i) for i
                                         in xrange(self.n_batches[2])]
                    now_loss = np.mean(test_loss)
                    print('After epoch %i, minibatch %i/%i, test error %f %%' % \
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
                    else:
                        print('Fuck, now loss>best loss, best loss is %f %%, patience now is %i' %\
                              (best_loss*100. ,self.patience) )

                if self.patience <= iter:
                    done_looping = True
                    break

