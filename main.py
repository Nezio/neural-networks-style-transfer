import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import numpy as np

import config

def main():
    # Tensorflow fix for the GPU memory allcation issue
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print_log("Physical GPUs: {p_gpu}, Logical GPUs: {l_gpu}".format(p_gpu=len(gpus), l_gpu=len(logical_gpus)))
        except RuntimeError as ex:
            # Memory growth must be set before GPUs have been initialized
            print_log(ex)

    # instantiate network architecture (model)
    external_model = vgg19.VGG19(include_top=False, weights="imagenet")

    # get all layer names and outputs from the model
    external_model_layers = {}
    for layer in external_model.layers:
        external_model_layers[layer.name] = layer.output

    # create model (based on external_model) that returns all the activations
    model = keras.Model(inputs=external_model.inputs, outputs=external_model_layers)

    # set optimizer
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    # set generated image size
    content_image_width, content_image_height = keras.preprocessing.image.load_img(config.content_image_path).size
    image_width = config.image_width
    image_height = int(image_width / content_image_width * content_image_height)

    # load and preprocess images
    content_image = load_image(config.content_image_path, image_width, image_height)
    style_image = load_image(config.style_image_path, image_width, image_height)
    generated_image = tf.Variable(load_image(config.generated_image_path, image_width, image_height))

    # TODO: try getting activations for content and style image before the loop

    # training loop
    for i in range(1, config.iterations + 1):
        # get loss and gradients
        loss, gradients = get_loss_and_gradients(content_image, style_image, generated_image, model)

        # optimize
        # log and output

    # final results
    x = 42

##############################################################################################################################################
##############################################################################################################################################
        
def print_log(text):
    print("[" + datetime.datetime.now().strftime("%Y.%m.%d. %H:%M:%S") + "]", end="")
    print(" " + text)

def load_image(image_path, image_width, image_height):
    image = keras.preprocessing.image.load_img(image_path, target_size=(image_width, image_height))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = vgg19.preprocess_input(image)
    image = tf.convert_to_tensor(image)

    return image

#@tf.function # Compiles a function into a callable TensorFlow graph.
def get_loss_and_gradients(content_image, style_image, generated_image, model):
    with tf.GradientTape() as gt:
        loss = get_loss(content_image, style_image, generated_image, model)

    gradients = gt.gradient(loss, generated_image)

    return loss, gradients

def get_loss(content_image, style_image, generated_image, model):
    
    # get all activations
    input_tensor = tf.concat([content_image, style_image, generated_image], axis=0)
    all_activations = model(input_tensor)

    # calculate content loss
    content_layer_activations = all_activations[config.content_layer_name]
    content_image_activations = content_layer_activations[0,:,:,:]
    generated_image_activations = content_layer_activations[2,:,:,:]
    content_loss = config.content_weight * get_content_loss(content_image_activations, generated_image_activations)

    # calculate style loss
    style_loss = tf.zeros(shape=())
    style_weight = config.style_weight / len(config.style_layer_names)
    image_width = content_image.shape[1]
    image_height = content_image.shape[2]
    for style_layer_name in config.style_layer_names:
        style_layer_activations = all_activations[style_layer_name]
        style_image_activations = style_layer_activations[1,:,:,:]
        generated_image_activations = style_layer_activations[2,:,:,:]
        style_loss += style_weight * get_style_loss(content_image_activations, generated_image_activations, image_width, image_height)

    # calculate total variation loss

    # calculate final loss

    return loss

def get_content_loss(content_image_activations, generated_image_activations):
    # TODO: added *0.5; check if it produces worse output image
    loss = 0.5 * tf.reduce_sum(tf.square(generated_image_activations - content_image_activations))

    return loss

def get_style_loss(content_image_activations, generated_image_activations, image_width, image_height):
    
    x = 12


if __name__ == "__main__":
    main()