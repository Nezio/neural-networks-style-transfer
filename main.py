import os
import math
import time
import datetime
import shutil
from PIL import Image
import re

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

    while (True):
        # initialize/clear image paths
        content_image_path = None
        style_image_path = None
        generated_image_path = None

        # check input folder for subfolders with images
        input_subfolders = [f.path for f in os.scandir(config.input_folder) if f.is_dir()]
        if len(input_subfolders) > 0 and len([f.path for f in os.scandir(input_subfolders[0])]) >= 2:
            # get image paths from the first subfolder (process oldest first; FIFO)
            subfolder0_images = [f.path for f in os.scandir(input_subfolders[0])]
            content_image_path = [s for s in subfolder0_images if "content" in s][0]
            style_image_path = [s for s in subfolder0_images if "style" in s][0]

            # replace slashes
            content_image_path = content_image_path.replace("\\", "/")
            style_image_path = style_image_path.replace("\\", "/" )
            
            generated_image_path = content_image_path
        else:
            # nothing found yet; get some sleep
            time.sleep(1.0)
            continue

        # create output subfolder structure
        job_result_folder = None
        output_input_folder = None
        input_folder_name = input_subfolders[0].split("/")[-1]
        if re.search(".+_.+_.+", input_subfolders[0]):
            # create output subfolder structure (web app folder)

            output_folder_datetime = input_folder_name.split("_")[0]
            output_folder_userid = input_folder_name.split("_")[1]
            output_folder_jobid = input_folder_name.split("_")[2]
            
            # create user folder if it doesn't already exist
            user_folder = os.path.join(config.output_folder, output_folder_userid)
            if not os.path.exists(user_folder):
                os.mkdir(user_folder)

            # create job folder if it doesn't already exist
            job_result_folder = os.path.join(user_folder, output_folder_datetime + "_" + output_folder_jobid)
            if not os.path.exists(job_result_folder):
                os.mkdir(job_result_folder)
            else:
                # this shouldn't be possible during normal execution
                raise Exception("Job folder {folder} already exists! Did you stop this script before the cleanup section the last time it ran?".format(folder=job_result_folder))

            # create input folder within the output job folder
            output_input_folder = os.path.join(job_result_folder, "input")
            if not os.path.exists(output_input_folder):
                os.mkdir(output_input_folder)
        else:
            # create output subfolder structure (regular folder)

            # create job folder; remove old one if it exists
            job_result_folder = os.path.join(config.output_folder, input_folder_name)
            if os.path.exists(job_result_folder):
                shutil.rmtree(job_result_folder)
            if not os.path.exists(job_result_folder):
                os.mkdir(job_result_folder)
            
            # create input folder within the output job folder
            output_input_folder = os.path.join(job_result_folder, "input")
            if not os.path.exists(output_input_folder):
                os.mkdir(output_input_folder)


        # set generated image size
        content_image_width, content_image_height = keras.preprocessing.image.load_img(content_image_path).size
        if content_image_width >= content_image_height:
            # landscape
            image_width = config.image_long_side
            image_height = int(config.image_long_side / content_image_width * content_image_height)
        else:
            # portrait
            image_height = config.image_long_side
            image_width = int(config.image_long_side / content_image_height * content_image_width)

        print_log("Total number of pixels: {pixels}".format(pixels=str(image_height * image_width)))

        # load and preprocess images
        content_image = load_image(content_image_path, image_width, image_height)
        style_image = load_image(style_image_path, image_width, image_height)
        generated_image = tf.Variable(load_image(generated_image_path, image_width, image_height))

        # update image size in case it was cropped
        image_width = generated_image.shape[2]
        image_height = generated_image.shape[1]

        # measure time
        start_time = time.time()
        initialization_time = None

        # initialize estimate time
        estimate_start_time = None

        # get all activations for content and style image
        input_tensor = tf.concat([content_image, style_image], axis=0)
        all_activations_content_style = model(input_tensor)

        # set optimizer
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

        # style transfer loop
        print_log("Starting style transfer with {i} iterations.".format(i=config.iterations))
        for i in range(1, config.iterations + 1):
            # get loss and gradients
            loss, gradients = get_loss_and_gradients(generated_image, all_activations_content_style, model)

            # optimize
            optimizer.apply_gradients([(gradients, generated_image)])

            # log and output
            if i % config.WIP_save_step == 0:
                print_log("Iteration {i}/{iterations}: loss={loss}".format(i=i, iterations=config.iterations, loss=str(round(loss.numpy(), 4))))
                padded_iteration_number = str(i).zfill(4)
                image_name = "output_" + padded_iteration_number + ".png"
                image_path = os.path.join(job_result_folder, image_name)
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
                print_log("Estimated time per iteration: {time}s".format(time=round(time_per_iteration, 4)))
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
        
        # move processed input subfolder to output subfolder and delete input subfolder
        move_all_files(input_subfolders[0], output_input_folder)
        os.rmdir(input_subfolders[0])

        # final logs
        print_log("Style transfer loop complete after {time}".format(time=total_time_str))
        print_log("Number of iterations done: {i}".format(i=config.iterations))
        print_log("Average time per iteration (not counting initialization time of {init_time}): {time} s".format(init_time=initialization_time_str, time=str(round(average_time_per_iteration, 4))))
        print_log("\n", include_timestamp=False)
    

##############################################################################################################################################
##############################################################################################################################################
        
def print_log(text, end="\n", include_timestamp=True):
    if(include_timestamp):
        print("[" + datetime.datetime.now().strftime("%Y.%m.%d. %H:%M:%S") + "]", end=" ")
    print(text, end=end)

def load_image(image_path, image_width, image_height):
    image = keras.preprocessing.image.load_img(image_path, target_size=(image_height, image_width))

    if image_width * image_height > config.max_pixel_count:
        # crop the image
        image = crop_image(image, 32, 19)

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
    

@tf.function # compiles a function into a callable TensorFlow graph
def get_loss_and_gradients(generated_image, all_activations_content_style, model):
    with tf.GradientTape() as gt:
        loss = get_loss(generated_image, all_activations_content_style, model)

    gradients = gt.gradient(loss, generated_image)

    return loss, gradients

def get_loss(generated_image, all_activations_content_style, model):
    
    # get all activations
    input_tensor = tf.concat([generated_image], axis=0)
    all_activations_generated = model(input_tensor)

    # calculate content loss
    content_image_activations = all_activations_content_style[config.content_layer_name][0,:,:,:]
    generated_image_activations = all_activations_generated[config.content_layer_name][0,:,:,:]
    content_loss = config.content_weight * get_content_loss(content_image_activations, generated_image_activations)

    # calculate style loss
    style_loss = tf.zeros(shape=())
    style_weight = config.style_weight / len(config.style_layer_names)
    image_width = generated_image.shape[2]
    image_height = generated_image.shape[1]
    image_size = image_width * image_height
    for style_layer_name in config.style_layer_names:
        style_image_activations = all_activations_content_style[style_layer_name][1,:,:,:]
        generated_image_activations = all_activations_generated[style_layer_name][0,:,:,:]
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

def crop_image(image, aspect_width, aspect_height):
    result_image = None
    offset = None

    if image.width < image.height:
        # portrait (landscape is default)
        aspect_width, aspect_height = aspect_height, aspect_width
    
    new_height = int(aspect_height / aspect_width * image.width)

    if new_height < image.height:
        # align width and clip top and bottom
        offset = math.floor((image.height - new_height) / 2)
        left = 0
        top = 0 + offset
        right = image.width
        bottom = image.height - offset
    else:
        # align hight and clip sides
        new_width = int(aspect_width / aspect_height * image.height)
        if new_width <= image.width:
            offset = math.floor((image.width - new_width) / 2)
        else:
            # this shouldn't even be possible
            raise Exception("New image width is somehow bigger!")
        left = offset
        top = 0
        right = image.width - offset
        bottom = image.height
    
    result_image = image.crop((left, top, right, bottom))

    return result_image

def move_all_files(source_folder, target_folder):
        
    file_names = os.listdir(source_folder)
        
    for file_name in file_names:
        shutil.move(os.path.join(source_folder, file_name), target_folder)
    






if __name__ == "__main__":
    main()