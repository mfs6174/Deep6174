#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: training_policy.py
# Date: Wed Sep 17 17:08:32 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

class TrainForever(object):

    def __init__(self, train_model, test_model,
                 n_batch, test_freq,
                 logger, learning_rate_provider):
        self.train_model = train_model
        self.test_model = test_model
        self.n_batch = n_batch
        self.test_freq = test_freq
        self.logger = logger
        self.learning_rate_provider = learning_rate_provider

    def work(self):
        best_loss = numpy.inf
        epoch = 0
        progressor = Progressor(None)

        while True:
            epoch = epoch + 1
            if epoch > 1: progressor.report(1, True)

            # save params at the beginning of each epoch
            logger.save_params(epoch)

            learning_rate = rate_provider.get_rate(epoch)
            print "In epoch {0}: learning rate is {1}".format(epoch, learning_rate)
            for minibatch_index in xrange(self.n_batch):
                iter = (epoch - 1) * self.n_batch + minibatch_index

                if iter % 200 == 0 or (iter % 10 == 0 and iter < 30) or (iter < 5):
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index, learning_rate)


                if (iter + 1) % self.test_freq == 0 or iter in [5, 10, 20, 30, 60, 100, 200]:
                    # do a validation:

                    # compute zero-one loss on validation set
                    test_loss = [test_model(i) for i
                                         in xrange(self.n_batch)]
                    now_loss = numpy.mean(test_loss)
                    print('After epoch %i, minibatch %i/%i, test error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_batch, \
                           now_loss * 100.))

                    # got the best score until now
                    if now_loss < best_loss:
                        best_loss = now_loss
                        if best_loss < 0.85:
                            # save best params
                            print 'Yay! Saving best model ...'
                            logger.save_params('best')
