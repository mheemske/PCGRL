import tensorflow as tf

class BatchRenormalization(tf.keras.layers.Layer):

    def __init__(self, alpha=0.01, rmax=1.5, dmax=2, epsilon=10**-6, **kwargs):
        super(BatchRenormalization, self).__init__()

        self.alpha = alpha
        self.rmax = rmax
        self.dmax = dmax
        self.epsilon = epsilon

        self.gamma = self.add_weight(name='gamma',
                                     shape=(1,),
                                     initializer='ones',
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=(1,),
                                    initializer='zeros',
                                    trainable=True)

    def build(self, input_shape):
        self.mu = tf.Variable(tf.zeros(input_shape[1:]), "mu")
        self.sigma = tf.Variable(tf.ones(input_shape[1:]), "sigma")

    def call(self, x, training=False):
        if training:
            batch_mu = tf.reduce_mean(x, 0)
            batch_sigma = tf.sqrt(self.epsilon + tf.reduce_mean(tf.square(x - batch_mu), 0))

            r = tf.stop_gradient(tf.clip_by_value(batch_sigma / self.sigma, 1 / self.rmax, self.rmax))
            d = tf.stop_gradient(tf.clip_by_value((batch_mu - self.mu) / self.sigma, -self.dmax, self.dmax))

            self.mu.assign_add(self.alpha * (batch_mu - self.mu))
            self.sigma.assign_add(self.alpha * (batch_sigma - self.sigma))

            _x = r * (x - batch_mu) / batch_sigma + d
        else:
            _x = (x - self.mu) / self.sigma
        
        return self.gamma * _x + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
