# Mined Hackathon (Diamond Quality Analysis and Background Removal)

Our approach employs a pre-trained U-2 Net alongside auxiliary ensemble networks, designed to operate seamlessly with the latest TensorFlow and Numpy versions. This repository serves as the final submission by Team DetrousEN for the MineD Hackathon, where the focal task centered on background and foreground segmentation.

Additionally, we've developed a Python package equipped with command-line functionalities to streamline testing strategies and facilitate rapid deployment.

Within the 'Ensemble' folder, you'll find a Jupyter notebook dedicated to the ensemble networks. Here, the 'AND' function serves as a key element in combining the underlying encoders. Our ensemble network comprises a SqueezeUnet, UNet, U-2 Net, and U-2 Net Plus. For testing purposes on 256 images, we propose using a standard U-2 Net. The segmented images resulting from these tests can be accessed [here](https://drive.google.com/drive/folders/1nX8PM-ECYulucVZ16eEtvKVDzP7Z38FW?usp=sharing).

All pre-trained networks are conveniently accessible [here](https://drive.google.com/drive/folders/1CT7rw7tyGVCazT-ij9rAnLCBM7s_o-Ax).

For a comprehensive overview, we've also provided a video presentation, which can be viewed [here](https://drive.google.com/file/d/1AhN9Ywjk5c9YpPGrWC3LLGukKO0HqLIC/view?usp=share_link).
