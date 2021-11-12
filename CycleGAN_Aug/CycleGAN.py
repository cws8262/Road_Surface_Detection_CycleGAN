import tensorflow as tf
import tensorflow_addons as tfa


class discriminator(tf.keras.Model):
    def __init__(self, input_shape = (None, 256,256,3), **kwargs):
        super(discriminator, self).__init__(**kwargs)
        LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        self.C64 = tf.keras.layers.Conv2D(64, (4,4), padding = 'same', 
                                          strides = 2,
                                          activation = LeakyReLU)
        self.C128 = cglayer_Ck(128, 2, 'same')
        self.C256 = cglayer_Ck(256, 2, 'same')
        self.C512 = cglayer_Ck(512, 1, 'valid')
        self.C1 = tf.keras.layers.Conv2D(1, (4,4), padding = 'valid')

        self.build(input_shape=input_shape)
        self.call(tf.keras.Input(input_shape[1:]))
        
    def call(self, inputs, training = False):
        x = self.C64(inputs)
        x = self.C128(x, training)
        x = self.C256(x, training)
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],"SYMMETRIC")
        x = self.C512(x, training)
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],"SYMMETRIC")
        x = self.C1(x)
        
        return x
        
class cglayer_Ck(tf.keras.layers.Layer):
    def __init__(self, num_filter, strides, padding):
        super(cglayer_Ck, self).__init__()      
        self.LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)  
        self.conv2d = tf.keras.layers.Conv2D(num_filter, (4,4),
                                             padding = padding,
                                             strides = strides)
        self.insn = tfa.layers.InstanceNormalization()
    
    def call(self, inputs, training=False):
        
        x = self.conv2d(inputs)
        x = self.insn(x, training = training)
        x = self.LeakyReLU(x)
        
        return x
    
class generator(tf.keras.Model):
    def __init__(self, input_shape = (None, 256,256,3), **kwargs):
        super(generator, self).__init__(**kwargs)
        
        self.c7s1_64 = cglayer_c7s1_k(64)
        self.d128 = cglayer_dk(128)
        self.d256 = cglayer_dk(256)
        self.R256_1 = cglayer_Rk(256)
        self.R256_2 = cglayer_Rk(256)
        self.R256_3 = cglayer_Rk(256)
        self.R256_4 = cglayer_Rk(256)
        self.R256_5 = cglayer_Rk(256)
        self.R256_6 = cglayer_Rk(256)
        self.R256_7 = cglayer_Rk(256)
        self.R256_8 = cglayer_Rk(256)
        self.R256_9 = cglayer_Rk(256)
        self.u128 = cglayer_uk(128)
        self.u64 = cglayer_uk(64)
        self.c7s1_3 = cglayer_c7s1_k(3)
        
        self.build(input_shape=input_shape)
        self.call(tf.keras.Input(input_shape[1:]))
        
    def call(self, inputs, training = False):
        x = self.c7s1_64(inputs, training)
        x = tf.nn.relu(x)
        x = self.d128(x, training)
        x = self.d256(x, training)
        x = self.R256_1(x, training)
        x = self.R256_2(x, training)
        x = self.R256_3(x, training)
        x = self.R256_4(x, training)
        x = self.R256_5(x, training)
        x = self.R256_6(x, training)
        x = self.R256_7(x, training)
        x = self.R256_8(x, training)
        x = self.R256_9(x, training)
        x = self.u128(x)
        x = self.u64(x)
        x = self.c7s1_3(x)
        x = tf.nn.tanh(x)
        return x

class cglayer_Rk(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(cglayer_Rk, self).__init__()        
        self.bn_a = tf.keras.layers.BatchNormalization()
        self.bn_b = tf.keras.layers.BatchNormalization()
        
        self.conv2d_a = tf.keras.layers.Conv2D(num_filter, 3, padding='same')
        self.conv2d_b = tf.keras.layers.Conv2D(num_filter, 3, padding='same')
        self.conv2d_c = tf.keras.layers.Conv2D(num_filter, 1, padding='same')
    
    def call(self, inputs, training=False):
        
        x = self.conv2d_a(inputs)
        x = self.bn_a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2d_b(x)
        x = self.bn_b(x, training=training)
        
        x = x + self.conv2d_c(inputs)
        x = tf.nn.relu(x)
        
        return x

class cglayer_c7s1_k(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(cglayer_c7s1_k, self).__init__()     
        self.conv2d = tf.keras.layers.Conv2D(num_filter, 7, padding='same')
        self.insn = tfa.layers.InstanceNormalization()
        
    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.insn(x, training = training)
        return x
    
class cglayer_dk(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(cglayer_dk, self).__init__()     
        self.conv2d = tf.keras.layers.Conv2D(num_filter, 3, padding='same',
                                             strides = 2)
        self.insn = tfa.layers.InstanceNormalization()
        
    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.insn(x, training = training)
        x = tf.nn.relu(x)
        return x
    
class cglayer_uk(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(cglayer_uk, self).__init__()  
        self.conv2d = tf.keras.layers.Conv2DTranspose(num_filter, 3, 
                                                      padding='same',
                                                      strides = 2)
        self.insn = tfa.layers.InstanceNormalization()
        
    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.insn(x, training = training)
        x = tf.nn.relu(x)
        return x    
