# input folder (this folder will be checked for subfolders; each subfolder should contain a content and a style image named "content" and "style")
input_folder = "input/"

# output folder (inside of this folder subfolders will be created, one for each input subfolder)
output_folder = "output/"


# long side of the generated (output) image (e.g.: if image is landscape long side is image width); the short side is calculated automatically
image_long_side = 560

# maximum number of pixels per image that the network can process (determined heuristically)
max_pixel_count = 186000


# number of iterations to run
iterations = 1000

# after how many iterations should WIP image be saved
WIP_save_step = 100

# number of iterations to use in estimate calculation
estimate_iterations = 30


# learning rate
learning_rate = 10

# name of a layer from pre-trained CNN to extract content activations from
content_layer_name = "block5_conv2"

# names of layers from pre-trained CNN to extract style activations from
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]


# content weight (alpha)
content_weight = 0.0000001

# style weight (beta)
style_weight = 0.0000001

# total variation weight
total_variation_weight = 0.0000001