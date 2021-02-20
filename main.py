import time
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

    # measure time
    start_time = time.time()
    initialization_time = None

    # initialize estimate time
    estimate_start_time = None

    # style transfer loop
    print_log("Starting style transfer with {i} iterations.".format(i=config.iterations))
    for i in range(1, config.iterations + 1):
        # get loss and gradients
        loss, gradients = get_loss_and_gradients(content_image, style_image, generated_image, model)

        # optimize
        optimizer.apply_gradients([(gradients, generated_image)])

        # log and output
        if i % config.WIP_save_step == 0:
            print_log("Iteration {i}/{iterations}: loss={loss}".format(i=i, iterations=config.iterations, loss=str(round(loss.numpy(), 4))))
            image_path = "output/output_" + str(i) + ".png"
            output_image = save_image(generated_image, image_path, image_width, image_height)
        else:
            print_log("*", end="", include_timestamp=False)

        # calculate estimate (once)
        if i == 1:
            estimate_start_time = time.time()
            initialization_time = estimate_start_time - start_time
        if i == config.estimate_iterations + 1:
            estimate_end_time = time.time()
            estimate_time_i_iterations = int(estimate_end_time - estimate_start_time)
            time_per_iteration = estimate_time_i_iterations / config.estimate_iterations
            total_estimated_time_remaining = time_per_iteration * config.iterations
            total_estimated_time_remaining_str = str(datetime.timedelta(seconds=round(total_estimated_time_remaining, 4)))
            print_log("", include_timestamp=False)
            print_log("Time per iteration: {time}s".format(time=round(time_per_iteration, 4)))
            print_log("Estimated time remaining: {time}".format(time=total_estimated_time_remaining_str))
            for j in range(config.estimate_iterations + 1):
                print_log("*", end="", include_timestamp=False)
            estimate_calculated = True

    # measure time
    end_time = time.time()
    total_time = int(end_time - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    average_time_per_iteration = (total_time - initialization_time) / config.iterations
    initialization_time_str = str(datetime.timedelta(seconds=round(initialization_time, 4)))
    
    print_log("Style transfer loop complete after {time}".format(time=total_time_str))
    print_log("Number of iterations done: {i}".format(i=config.iterations))
    print_log("Average time per iteration (not counting initialization time ({init_time})): {time} s".format(init_time=initialization_time_str, time=str(round(average_time_per_iteration, 4))))
    

##############################################################################################################################################
##############################################################################################################################################
        
def print_log(text, end="\n", include_timestamp=True):
    if(include_timestamp):
        print("[" + datetime.datetime.now().strftime("%Y.%m.%d. %H:%M:%S") + "]", end=" ")
    print(text, end=end)

def load_image(image_path, image_width, image_height):
    image = keras.preprocessing.image.load_img(image_path, target_size=(image_height, image_width))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = vgg19.preprocess_input(image)
    image = tf.convert_to_tensor(image)

    return image

def save_image(image, path, image_width, image_height):
    image = image.numpy()
    image = image.reshape((image_height, image_width, 3))

    # remove zero-center by mean pixel
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:,:, 2] += 123.68
    
    # BGR to RGB
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype("uint8")

    # save the image
    keras.preprocessing.image.save_img(path, image)
    

@tf.function # Compiles a function into a callable TensorFlow graph.
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
    image_width = content_image.shape[2]
    image_height = content_image.shape[1]
    image_size = image_width * image_height
    for style_layer_name in config.style_layer_names:
        style_layer_activations = all_activations[style_layer_name]
        style_image_activations = style_layer_activations[1,:,:,:]
        generated_image_activations = style_layer_activations[2,:,:,:]
        style_loss += style_weight * get_style_loss(style_image_activations, generated_image_activations, image_size)

    # calculate total variation loss
    total_variation_loss = config.total_variation_weight * get_total_variation_loss(generated_image, image_width, image_height)

    # calculate final loss
    loss = content_loss + style_loss + total_variation_loss

    return loss

def get_content_loss(content_image_activations, generated_image_activations):
    loss = 0.5 * tf.reduce_sum(tf.square(generated_image_activations - content_image_activations))

    return loss

def get_style_loss(style_image_activations, generated_image_activations, image_size):
    # calculate gram matrices
    gram_matrix_style = get_gram_matrix(style_image_activations)
    gram_matrix_generated = get_gram_matrix(generated_image_activations)

    # calculate style
    channels = 3

    style_loss = tf.reduce_sum(tf.square(gram_matrix_style - gram_matrix_generated)) / (4.0 * (channels ** 2) * (image_size ** 2))

    return style_loss


def get_gram_matrix(activations):
    # transpose activations (e.g.: shape [512, 256, 64] to shape [64, 512, 256])
    transposed_activations = tf.transpose(activations, (2, 0, 1))

    # vectorize transposed activations (e.g.:  shape [64, 512, 256] to shape [64, 512*256] = [64, 131072])
    vectorized_activations = tf.reshape(transposed_activations, (tf.shape(transposed_activations)[0], -1))

    # calculate gram matrix    
    gram_matrix = tf.matmul(vectorized_activations, tf.transpose(vectorized_activations))
    
    return gram_matrix

def get_total_variation_loss(image, image_width, image_height):
    a = tf.square(image[:, : image_height - 1, : image_width - 1, :] - image[:, 1:, : image_width - 1, :])
    b = tf.square(image[:,:image_height - 1,:image_width - 1,:] - image[:,:image_height - 1, 1:,:])
    total_variation_loss = tf.reduce_sum(tf.pow(a + b, 1.25))

    return total_variation_loss






if __name__ == "__main__":
    main()