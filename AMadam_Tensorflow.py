
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
from six.moves import zip

from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces

from keras.optimizers import Optimizer

class AMadam(Optimizer):


    def __init__(self, lr=0.02, beta_1=0.9,
                 epsilon=None, decay=0., 
                 batch_size=1, samples_per_epoch=1, 
                 epochs=1, **kwargs):
        super(ImpAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1') 
            self.decay = K.variable(decay, name='decay')
            self.batch_size = K.variable(batch_size, name='batch_size')
            self.samples_per_epoch = K.variable(samples_per_epoch, name='samples_per_epoch')
            self.epochs = K.variable(epochs, name='epochs')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        '''Bias corrections according to the Adam paper
        '''
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_1, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        
        for p, g, m, v in zip(params, grads, ms, vs):
            
			#adam update for first momntum
            m_t = (self.beta_1 * m)+ (1. - self.beta_1) *(g)
			
			#adam update for second momntum
			#v_t = (self.beta_2 * v)+ (1. - self.beta_2) *(g^2)
			
			#AMadam update for second momntum
            v_t=K.abs(m_t)
			
           
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            
            self.updates.append(K.update(m, m_t))
			self.updates.append(K.update(v, v_t))
            
           
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'decay': float(K.get_value(self.decay)),
                  'batch_size': int(K.get_value(self.batch_size)),
                  'samples_per_epoch': int(K.get_value(self.samples_per_epoch)),
                  'epochs': int(K.get_value(self.epochs)),
                  'epsilon': self.epsilon}
        base_config = super(AMadam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




AMadam=AMadam(lr=0.01)    