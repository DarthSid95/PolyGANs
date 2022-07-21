
import tensorflow as tf

def ResBlockUp(inputs, output_channel):
	init_fn = tf.keras.initializers.glorot_uniform()
	init_fn = tf.function(init_fn, autograph=False)
	x = tf.keras.layers.BatchNormalization(momentum=0.9)(inputs)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = SpectralNormalization(tf.keras.layers.Conv2DTranspose(output_channel, 5, strides=2, kernel_initializer = init_fn, padding = 'SAME'))(x)
	x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = SpectralNormalization(tf.keras.layers.Conv2D(output_channel, 3, strides=1, padding = 'SAME'))(x)
	skip = SpectralNormalization(tf.keras.layers.Conv2DTranspose(output_channel, 3,strides=2,kernel_initializer = init_fn, padding = 'SAME'))(inputs)
	print(x , skip)
	x = x + skip
	return x


def ResBlockDown(inputs, output_channel):
	init_fn = tf.keras.initializers.glorot_uniform()
	init_fn = tf.function(init_fn, autograph=False)
	x = tf.keras.layers.BatchNormalization(momentum=0.9)(inputs)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = SpectralNormalization(tf.keras.layers.Conv2D(output_channel, 5, strides=2, kernel_initializer = init_fn, padding = 'SAME'))(x)
	x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = SpectralNormalization(tf.keras.layers.Conv2D(output_channel, 3, strides=1, padding = 'SAME'))(x)
	skip = SpectralNormalization(tf.keras.layers.Conv2D(output_channel, 3, strides=2, kernel_initializer = init_fn, padding = 'SAME'))(inputs)
	x = x + skip
	return x


class SpectralNormalization(tf.keras.layers.Wrapper):
	def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
		self.iteration = iteration
		self.eps = eps
		self.do_power_iteration = training
		if not isinstance(layer, tf.keras.layers.Layer):
			raise ValueError(
				'Please initialize `TimeDistributed` layer with a '
				'`Layer` instance. You passed: {input}'.format(input=layer))
		super(SpectralNormalization, self).__init__(layer, **kwargs)

	def build(self, input_shape):
		self.layer.build(input_shape)

		self.w = self.layer.kernel
		self.w_shape = self.w.shape.as_list()

		self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
								 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
								 trainable=False,
								 name='sn_v',
								 dtype=tf.float32)

		self.u = self.add_weight(shape=(1, self.w_shape[-1]),
								 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
								 trainable=False,
								 name='sn_u',
								 dtype=tf.float32)

		super(SpectralNormalization, self).build()

	def call(self, inputs):
		self.update_weights()
		output = self.layer(inputs)
		self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
		return output
	
	def update_weights(self):
		w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
		
		u_hat = self.u
		v_hat = self.v  # init v vector

		if self.do_power_iteration:
			for _ in range(self.iteration):
				v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
				v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

				u_ = tf.matmul(v_hat, w_reshaped)
				u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

		sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
		self.u.assign(u_hat)
		self.v.assign(v_hat)

		self.layer.kernel.assign(self.w / sigma)

	def restore_weights(self):
		self.layer.kernel.assign(self.w)



##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
# Regularization
##################################################################################

def orthogonal_regularizer(scale) :
	""" Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

	def ortho_reg(w) :
		""" Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
		_, _, _, c = w.get_shape().as_list()

		w = tf.reshape(w, [-1, c])

		""" Declaring a Identity Tensor of appropriate size"""
		identity = tf.eye(c)

		""" Regularizer Wt*W - I """
		w_transpose = tf.transpose(w)
		w_mul = tf.matmul(w_transpose, w)
		reg = tf.subtract(w_mul, identity)

		"""Calculating the Loss Obtained"""
		ortho_loss = tf.nn.l2_loss(reg)

		return scale * ortho_loss

	return ortho_reg

def orthogonal_regularizer_fully(scale) :
	""" Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

	def ortho_reg_fully(w) :
		""" Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
		_, c = w.get_shape().as_list()

		"""Declaring a Identity Tensor of appropriate size"""
		identity = tf.eye(c)
		w_transpose = tf.transpose(w)
		w_mul = tf.matmul(w_transpose, w)
		reg = tf.subtract(w_mul, identity)

		""" Calculating the Loss """
		ortho_loss = tf.nn.l2_loss(reg)

		return scale * ortho_loss

	return ortho_reg_fully

	

weight_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)
weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)

# Regularization only G in BigGAN

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
	with tf.name_scope(scope):
		if pad > 0:
			h = x.get_shape().as_list()[1]
			if h % stride == 0:
				pad = pad * 2
			else:
				pad = max(kernel - (h % stride), 0)

			pad_top = pad // 2
			pad_bottom = pad - pad_top
			pad_left = pad // 2
			pad_right = pad - pad_left

			if pad_type == 'zero' :
				x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
			if pad_type == 'reflect' :
				x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

		if sn :
			if scope.__contains__('generator') :
				w = tf.compat.v1.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
									regularizer=weight_regularizer)
			else :
				w = tf.compat.v1.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
									regularizer=None)

			x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
							 strides=[1, stride, stride, 1], padding='VALID')
			if use_bias :
				bias = tf.compat.v1.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
				x = tf.nn.bias_add(x, bias)

		else :
			if scope.__contains__('generator'):
				x = tf.compat.v1.layers.conv2d(inputs=x, filters=channels,
									 kernel_size=kernel, kernel_initializer=weight_init,
									 kernel_regularizer=weight_regularizer,
									 strides=stride, use_bias=use_bias)
			else :
				x = tf.compat.v1.layers.conv2d(inputs=x, filters=channels,
									 kernel_size=kernel, kernel_initializer=weight_init,
									 kernel_regularizer=None,
									 strides=stride, use_bias=use_bias)


		return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
	with tf.name_scope(scope):
		x_shape = x.get_shape().as_list()

		if padding == 'SAME':
			output_shape = [100, x_shape[1] * stride, x_shape[2] * stride, channels]

		else:
			output_shape =[100, x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

		if sn :
			w = tf.compat.v1.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
			x = tf.nn.conv2d_transpose(x, filters=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

			if use_bias :
				bias = tf.compat.v1.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
				x = tf.nn.bias_add(x, bias)

		else :
			x = tf.compat.v1.layers.conv2d_transpose(inputs=x, filters=channels,
										   kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
										   strides=stride, padding=padding, use_bias=use_bias)

		return x

def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
	with tf.name_scope(scope):
		x = flatten(x)
		shape = x.get_shape().as_list()
		channels = shape[-1]

		if sn :
			if scope.__contains__('generator'):
				w = tf.compat.v1.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init, regularizer=weight_regularizer_fully)
			else :
				w = tf.compat.v1.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init, regularizer=None)

			if use_bias :
				bias = tf.compat.v1.get_variable("bias", [units], initializer=tf.constant_initializer(0.0))

				x = tf.matmul(x, spectral_norm(w)) + bias
			else :
				x = tf.matmul(x, spectral_norm(w))

		else :
			if scope.__contains__('generator'):
				x = tf.compat.v1.layers.dense(x, units=units, kernel_initializer=weight_init,
									kernel_regularizer=weight_regularizer_fully, use_bias=use_bias)
			else :
				x = tf.compat.v1.layers.dense(x, units=units, kernel_initializer=weight_init,
									kernel_regularizer=None, use_bias=use_bias)

		return x

def flatten(x) :
	return tf.compat.v1.layers.flatten(x)

def hw_flatten(x) :
	return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

##################################################################################
# Residual-block, Self-Attention-block
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
	with tf.name_scope(scope):
		with tf.name_scope('res1'):
			x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
			x = batch_norm(x, is_training)
			x = relu(x)

		with tf.name_scope('res2'):
			x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
			x = batch_norm(x, is_training)

		return x + x_init

def resblock_up(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
	with tf.name_scope(scope):
		with tf.name_scope('res1'):
			x = batch_norm(x_init, is_training)
			x = relu(x)
			x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

		with tf.name_scope('res2') :
			x = batch_norm(x, is_training)
			x = relu(x)
			x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)

		with tf.name_scope('skip') :
			x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)


	return x + x_init

def resblock_up_condition(x_init, z, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
	with tf.name_scope(scope):
		with tf.name_scope('res1'):
			x = condition_batch_norm(x_init, z, is_training)
			x = relu(x)
			x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

		with tf.name_scope('res2') :
			x = condition_batch_norm(x, z, is_training)
			x = relu(x)
			x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)

		with tf.name_scope('skip') :
			x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)


	return x + x_init


def resblock_down(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_down'):
	with tf.name_scope(scope):
		with tf.name_scope('res1'):
			x = batch_norm(x_init, is_training)
			x = relu(x)
			x = conv(x, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)

		with tf.name_scope('res2') :
			x = batch_norm(x, is_training)
			x = relu(x)
			x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

		with tf.name_scope('skip') :
			x_init = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)


	return x + x_init

def self_attention(x, channels, sn=False, scope='self_attention'):
	with tf.name_scope(scope):
		f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
		g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
		h = conv(x, channels, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

		# N = h * w
		s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

		beta = tf.nn.softmax(s)  # attention map

		o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
		gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

		o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
		x = gamma * o + x

	return x

def self_attention_2(x, channels, sn=False, scope='self_attention'):
	with tf.name_scope(scope):
		f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
		f = max_pooling(f)

		g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']

		h = conv(x, channels // 2, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]
		h = max_pooling(h)

		# N = h * w
		s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

		beta = tf.nn.softmax(s)  # attention map

		o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
		gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

		o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
		o = conv(o, channels, kernel=1, stride=1, sn=sn, scope='attn_conv')
		x = gamma * o + x

	return x

##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
	gap = tf.reduce_mean(x, axis=[1, 2])

	return gap

def global_sum_pooling(x) :
	gsp = tf.reduce_sum(x, axis=[1, 2])

	return gsp

def max_pooling(x) :
	x = tf.compat.v1.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
	return x

def up_sample(x, scale_factor=2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
	return tf.nn.leaky_relu(x, alpha)


def relu(x):
	return tf.nn.relu(x)


def tanh(x):
	return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
	return tf.compat.v1.layers.batch_normalization(x,
										 momentum=0.9,
										 epsilon=1e-05,
										 training=is_training,
										 name=scope)

def condition_batch_norm(x, z, is_training=True, scope='batch_norm'):
	with tf.name_scope(scope) :
		_, _, _, c = x.get_shape().as_list()
		decay = 0.9
		epsilon = 1e-05

		test_mean = tf.compat.v1.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
		test_var = tf.compat.v1.get_variable("pop_var", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

		beta = fully_conneted(z, units=c, scope='beta')
		gamma = fully_conneted(z, units=c, scope='gamma')

		beta = tf.reshape(beta, shape=[-1, 1, 1, c])
		gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

		if is_training:
			batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
			ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
			ema_var = tf.assign(test_var, test_var * decay + batch_var * (1 - decay))

			with tf.control_dependencies([ema_mean, ema_var]):
				return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
		else:
			return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)


def spectral_norm(w, iteration=1):
	w_shape = w.shape.as_list()
	w = tf.reshape(w, [-1, w_shape[-1]])

	u = tf.compat.v1.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u
	v_hat = None
	for i in range(iteration):
		"""
		power iteration
		Usually iteration = 1 will be enough
		"""

		v_ = tf.matmul(u_hat, tf.transpose(w))
		v_hat = tf.nn.l2_normalize(v_)

		u_ = tf.matmul(v_hat, w)
		u_hat = tf.nn.l2_normalize(u_)

	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

	with tf.control_dependencies([u.assign(u_hat)]):
		w_norm = w / sigma
		w_norm = tf.reshape(w_norm, w_shape)

	return w_norm