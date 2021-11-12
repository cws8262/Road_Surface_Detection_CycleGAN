import numpy as np
import tensorflow as tf
from PIL import Image
import os

tf.keras.backend.clear_session()

AUTOTUNE = tf.data.experimental.AUTOTUNE

WET_OR_SNOWY = True #

ld_dry = os.listdir('Image/dry')
ld_wet = os.listdir('Image/wet')
ld_snowy = os.listdir('Image/snowy')

dry_img = np.zeros((len(ld_dry), 256, 256, 3))
for i in range(len(ld_dry)):
    img = Image.open('Image/dry/'+ld_dry[i])
    dry_img[i,:] = np.array(img)
    
wet_img = np.zeros((len(ld_wet), 256, 256, 3))
for i in range(len(ld_wet)):
    img = Image.open('Image/wet/'+ld_wet[i])
    wet_img[i,:] = np.array(img)
    
snowy_img = np.zeros((len(ld_snowy), 256, 256, 3))
for i in range(len(ld_snowy)):
    img = Image.open('Image/snowy/'+ld_snowy[i])
    snowy_img[i,:] = np.array(img)
#%%
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

dry_ds = tf.data.Dataset.from_tensor_slices(dry_img)
wet_ds = tf.data.Dataset.from_tensor_slices(wet_img)
snowy_ds = tf.data.Dataset.from_tensor_slices(snowy_img)


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    
    return cropped_image
# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

ds_A = dry_ds.map(
    preprocess_image, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)
if WET_OR_SNOWY:
    ds_B = wet_ds.map(
        preprocess_image, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
else:
    ds_B = snowy_ds.map(
        preprocess_image, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
        
#%%
import CycleGAN
g_a2b = CycleGAN.generator()
g_b2a = CycleGAN.generator()
d_a = CycleGAN.discriminator()
d_b = CycleGAN.discriminator()

g_a2b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
g_b2a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

d_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
d_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)
def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss
#%%
import matplotlib.pyplot as plt
EPOCHS = 200

def generate_images(model_and_input1, model_and_input2):
    
    model1 = model_and_input1[0]
    input1 = model_and_input1[1]
    prediction1 = model1(input1)
    
    model2 = model_and_input2[0]
    input2 = model_and_input2[1]
    prediction2 = model2(input2)
    
    
    plt.figure(figsize=(12, 12))
    
    display_list = [input1[0], prediction1[0],\
                    input2[0], prediction2[0]]
    title = ['Input Image', 'Predicted Image',\
             'Input Image', 'Predicted Image']
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

# @tf.function
def train_step(real_a, real_b):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator a2b translates a -> b
        # Generator b2a translates b -> a.
      
        fake_b = g_a2b(real_a, training=True)
        cycled_a = g_b2a(fake_b, training=True)
      
        fake_a = g_b2a(real_b, training=True)
        cycled_b = g_a2b(fake_a, training=True)
      
        # same_x and same_y are used for identity loss.
        same_a = g_b2a(real_a, training=True)
        same_b = g_a2b(real_b, training=True)
      
        disc_real_a = d_a(real_a, training=True)
        disc_real_b = d_b(real_b, training=True)
      
        disc_fake_a = d_a(fake_a, training=True)
        disc_fake_b = d_b(fake_b, training=True)
      
        # calculate the loss
        g_a2b_loss = generator_loss(disc_fake_b)
        g_b2a_loss = generator_loss(disc_fake_a)
      
        total_cycle_loss = calc_cycle_loss(real_a, cycled_a) \
            + calc_cycle_loss(real_b, cycled_b)
      
        # Total generator loss = adversarial loss + cycle loss
        total_g_a2b_loss = g_a2b_loss + total_cycle_loss \
            + identity_loss(real_b, same_b)
        total_g_b2a_loss = g_b2a_loss + total_cycle_loss \
            + identity_loss(real_a, same_a)
      
        disc_a_loss = discriminator_loss(disc_real_a, disc_fake_a)
        disc_b_loss = discriminator_loss(disc_real_b, disc_fake_b)
    
    # Calculate the gradients for generator and discriminator
    g_a2b_gradients = tape.gradient(total_g_a2b_loss, 
                                          g_a2b.trainable_variables)
    g_b2a_gradients = tape.gradient(total_g_b2a_loss, 
                                          g_b2a.trainable_variables)
    
    discriminator_a_gradients = tape.gradient(disc_a_loss, 
                                              d_a.trainable_variables)
    discriminator_b_gradients = tape.gradient(disc_b_loss, 
                                              d_b.trainable_variables)
    
    # Apply the gradients to the optimizer
    g_a2b_optimizer.apply_gradients(zip(g_a2b_gradients, 
                                              g_a2b.trainable_variables))
    
    g_b2a_optimizer.apply_gradients(zip(g_b2a_gradients, 
                                              g_b2a.trainable_variables))
    
    d_a_optimizer.apply_gradients(zip(discriminator_a_gradients,
                                                  d_a.trainable_variables))
    
    d_b_optimizer.apply_gradients(zip(discriminator_b_gradients,
                                                  d_b.trainable_variables))
#%%
step_len = 1000

for epoch in range(EPOCHS):    
    n = 0
    a_iter = iter(ds_A)
    b_iter = iter(ds_B)
    for i in range(step_len):
        try:
            image_a = next(a_iter)
        except:
            a_iter = iter(ds_A)
            image_a = next(a_iter)
        try:
            image_b = next(b_iter)
        except:
            b_iter = iter(ds_B)
            image_b = next(b_iter)
        train_step(image_a, image_b)
        n+=1
        print("\r{:5}  {:5.2f}%".format(epoch, n/step_len * 100.),end='')
        
        if n%10 == 0:
            generate_images((g_a2b, image_a), (g_b2a, image_b))
    print()
#%%
from PIL import Image
# PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
if WET_OR_SNOWY:
    savepath = 'Image/wet_aug/wetaug'
else:
    savepath = 'Image/snowy_aug/snowyaug'
k = 0
for image_a in ds_A:
    k = k+1
    fake_b = ((g_a2b(image_a).numpy()[0,:]*0.5+0.5)*255).astype(np.uint8)
    aug = Image.fromarray(fake_b)
    
    aug.save(savepath+'{:05}'.format(k)+'.jpg')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        