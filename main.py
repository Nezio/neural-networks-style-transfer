import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import numpy as np

def main():
    # config
    content_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
    style_image_path = "images/wave.png"
    generated_image_path = content_image_path

    # generated image size
    width_content, height_content = keras.preprocessing.image.load_img(content_image_path).size
    image_width = 512
    image_height = int(image_width / width_content * height_content)

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
    optimizer = keras.optimizers.Adam(learning_rate=10)
    
    # load images
    content_image = preprocess_image(content_image_path, image_width, image_height)
    style_image = preprocess_image(style_image_path, image_width, image_height)
    generated_image = tf.Variable(preprocess_image(generated_image_path, image_width, image_height))

    # TODO: try getting activations for content and style image before the loop

    # training loop
        # loss
        # gradients
        # optimize
        # log and output

    # final results
    x = 42
        
def print_log(text):
    print("[" + datetime.datetime.now().strftime("%H:%M:%S") + "]", end="")
    print(" " + text)

def preprocess_image(image_path, image_width, image_height):
    image = keras.preprocessing.image.load_img(image_path, target_size=(image_width, image_height))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = vgg19.preprocess_input(image)
    image = tf.convert_to_tensor(image)
    return image




if __name__ == "__main__":
    main()