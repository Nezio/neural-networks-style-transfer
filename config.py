
# content image path
content_image_path = "images/#content.png"

# style image path
style_image_path = "images/#style.png"

# generated image path
generated_image_path = content_image_path

# width of the generated (output) image; the height is calculated automatically
image_width = 512


# learning rate
learning_rate = 10


# number of iterations to run
iterations = 3000

# after how many iterations should WIP image be saved
WIP_save_step = 100

# number of iterations to use in estimate calculation
estimate_iterations = 30


# name of a layer from pre-trained CNN to extract content activations from
content_layer_name = "block5_conv2"

# names of layers from pre-trained CNN to extract style activations from
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]


# content weight (alpha)
content_weight = 0.0000001

# style weight (beta)
style_weight = 0.0000001

# total variation weight
total_variation_weight = 0.0000001