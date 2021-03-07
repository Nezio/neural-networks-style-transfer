# Neural networks - style transfer

Implementation of style transfer algorithm using convolutional neural networks based on paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge. This project is done as a practical part of a master thesis for Faculty of Electronic Engineering.

Web-App part of this project: https://github.com/Nezio/neural-networks-style-transfer-web-app/

### Installation

- Install Python (developed on Python 3.8.7)
- Install tensorflow 2.4.0
- Install CUDA v 11.1.1 and CUDNN 8.0.4 (you can skip this if you want to use CPU instead of GPU, but running on GPU is highly recommended)
- Configure config.py
    Input and output folders need to be configured, the rest can remain as it is.
- Create a subfolder in the configured input folder and put two images inside, name them "content" and "style"
- Start the app by running main.py
- If you wish to configure this app to work with the web-app, you need to configure the input and output folders to be the corresponding folders in the web-app
The location in the web-app is .../StyleTransferWebApp/style_transfer_work_dir/input and .../StyleTransferWebApp/style_transfer_work_dir/output