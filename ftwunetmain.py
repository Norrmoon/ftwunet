#%%#####################################################################################################################
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
#%%#####################################################################################################################
## Generating Image and Mask Datasets ##
from tensorflow import keras
from pycocotools.coco import COCO
from random import shuffle
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import shutil
import rasterio
from skimage.transform import resize
from skimage.io import imread
#%%#####################################################################################################################
# Az aktuális felhasználó elérési útjának lekérdezése
user_path = os.path.expanduser("~")
#%%#####################################################################################################################
# Paths
base_path = os.path.expanduser("~/Downloads/ftw-dataset")
original_path = os.path.join(base_path, "original")
masks_path = os.path.join(base_path, "original_mask")
train_output_path = os.path.join(base_path, "train")
mask_train_output_path = os.path.join(base_path, "train_mask")
val_output_path = os.path.join(base_path, "val")
mask_val_output_path = os.path.join(base_path, "val_mask")
#%%#####################################################################################################################
# Ensure output directories exist
os.makedirs(train_output_path, exist_ok=True)
os.makedirs(mask_train_output_path, exist_ok=True)
os.makedirs(val_output_path, exist_ok=True)
os.makedirs(mask_val_output_path, exist_ok=True)

# Load file names from the original directory
file_names = [f for f in os.listdir(original_path) if f.endswith(".tif")]

# Shuffle the file names
random.shuffle(file_names)

# Split files
train_files = file_names[:3000]  # First 3000 files for training
val_files = file_names[3000:3300]  # Next 300 files for validation
#%%#####################################################################################################################
# Move train files
for file_name in train_files:
    shutil.move(os.path.join(original_path, file_name), os.path.join(train_output_path, file_name))
    shutil.move(os.path.join(masks_path, file_name), os.path.join(mask_train_output_path, file_name))

# Move validation files
for file_name in val_files:
    shutil.move(os.path.join(original_path, file_name), os.path.join(val_output_path, file_name))
    shutil.move(os.path.join(masks_path, file_name), os.path.join(mask_val_output_path, file_name))

print("Train and validation files successfully moved!")
#%%#####################################################################################################################
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, images_path, masks_path, batch_size):
        """
        CustomDataGenerator class for generating batches of preprocessed images and masks.

        Args:
            images_path (str): Path to the directory containing the original images.
            masks_path (str): Path to the directory containing the corresponding masks.
            batch_size (int): Number of samples in each batch.

        Attributes:
            images_path (str): Path to the directory containing the original images.
            masks_path (str): Path to the directory containing the corresponding masks.
            batch_size (int): Number of samples in each batch.
            image_filenames (list): List of matching filenames between images and masks.
            mask_filenames (list): List of matching filenames between masks and images.
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.image_filenames = self.get_matching_filenames()
        self.mask_filenames = self.get_matching_filenames()

    def get_matching_filenames(self):
        """
        Get the list of matching filenames between images and masks.

        Returns:
            list: List of matching filenames.
        """
        image_files = set([os.path.splitext(filename)[0] for filename in os.listdir(self.images_path)])
        mask_files = set([os.path.splitext(filename)[0] for filename in os.listdir(self.masks_path)])
        matching_files = list(image_files.intersection(mask_files))
        return matching_files
    
    def __len__(self):
        """
        Get the number of batches in the generator.

        Returns:
            int: Number of batches.
        """
        return int(np.ceil(len(self.image_filenames) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Get a batch of preprocessed images and masks.

        Args:
            idx (int): Batch index.

        Returns:
            tuple: Batch of preprocessed images and masks.
        """
        batch_filenames = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        batch_masks = []

        for filename in batch_filenames:
            image_path = os.path.join(self.images_path, filename + '.tif')
            mask_path = os.path.join(self.masks_path, filename + '.tif')

            # Load image and mask with Rasterio
            with rasterio.open(image_path) as src_image:
                image = src_image.read()  # Read all channels
            with rasterio.open(mask_path) as src_mask:
                mask = src_mask.read()  # Read the band of the mask

            image = np.moveaxis(image, 0, -1)
            mask = np.moveaxis(mask, 0, -1)
            mask = np.squeeze(mask)

            # Check if image and mask have the same dimensions
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Incompatible dimensions for image {image_path} and mask {mask_path}")

            # Resize the images and masks to size 128x128
            image = resize(image, (128, 128, image.shape[-1]), preserve_range=True, anti_aliasing=True)
            mask = resize(mask, (128, 128), preserve_range=True, anti_aliasing=False)

            # Convert the images and masks to arrays
            preprocessed_image = image
            preprocessed_mask = mask

            # Ellenőrizd, hogy a kép 4 csatornával rendelkezik-e
            if preprocessed_image.shape[-1] == 4:
                # Eltávolítjuk az alfa csatornát
                preprocessed_image = preprocessed_image[..., :3]

            # Check if image has 3 channels and shape of (128, 128, 3)
            if len(preprocessed_image.shape) == 3 and preprocessed_image.shape == (128, 128, 3):
                # Normalize the pixel values if needed
                preprocessed_image = preprocessed_image / 65535.0
                preprocessed_mask = (preprocessed_mask > 0).astype(np.float32)

                # Append the preprocessed images and masks to the batch
                batch_images.append(preprocessed_image)
                batch_masks.append(preprocessed_mask)
        
        # Convert the batch images and masks to numpy arrays and return
        return np.array(batch_images), np.array(batch_masks)
#%%#####################################################################################################################
# Usage

images_path = f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/train"
masks_path = f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/train_mask"
batch_size = 8


# Create an instance of the CustomDataGenerator
train_generator = CustomDataGenerator(images_path, masks_path, batch_size)
#%%#####################################################################################################################
def validate_image_shapes(generator):
    """
    Print the shapes of preprocessed images generated by the provided generator.

    Args:
        generator (CustomDataGenerator): Instance of the CustomDataGenerator class.
    """
    for i in range(len(generator)):
        # Get a batch of preprocessed images from the generator
        batch_images, _ = generator[i]
        
        # Print the shapes of the preprocessed images
        for image in batch_images:
            print(f"Shape of preprocessed image: {image.shape}")
            
validate_image_shapes(train_generator)
#%%#####################################################################################################################
# Print the number of files in the train directory containing original images
print(len(os.listdir(f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/train")))

# Print the number of files in the train_mask directory containing generated masks
print(len(os.listdir(f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/train_mask")))

# Print the number of files in the val directory containing original images
print(len(os.listdir(f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/val")))

# Print the number of files in the val_mask directory containing generated masks
print(len(os.listdir(f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/val_mask")))
#%%#####################################################################################################################
# Get the list of mask filenames
mask_filenames = [filename for filename in os.listdir(masks_path) if filename.endswith('.tif')]

# Assuming the first image in the mask folder is the one to plot
first_mask_filename = mask_filenames[0]
image_filename = os.path.splitext(first_mask_filename)[0] + '.tif'

# Load and plot the mask image
mask_image = imread(os.path.join(masks_path, first_mask_filename))
plt.subplot(1, 2, 1)
plt.imshow(mask_image, cmap='gray')  # Maszk esetén gyakran fekete-fehér
#plt.imshow(mask_image)
plt.title('Mask Image')
plt.axis('off')

# Load and plot the corresponding main image
main_image = imread(os.path.join(images_path, image_filename))
plt.subplot(1, 2, 2)
# Normalizálás és csatornaszám csökkentés
main_image = main_image[..., :3]  # Csak RGB csatornák
main_image = (main_image / np.max(main_image) * 255).astype(np.uint8)  # Átskálázás 0-255 közé
plt.imshow(main_image)
plt.title('Main Image')
plt.axis('off')

# Print the shapes of the images
print('Mask Image Shape:', mask_image.shape)
print('Main Image Shape:', main_image.shape)

plt.tight_layout()
plt.show()
#%%#####################################################################################################################
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
import keras
from keras.layers import*
from keras.optimizers import*
import pydot
import graphviz
#%%#####################################################################################################################
def down_block(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal",
    max_pool_window=(2, 2),
    max_pool_stride=(2, 2)
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    # # conv for skip connection
    conv = Activation("relu")(conv)

    pool = MaxPooling2D(pool_size=max_pool_window, strides=max_pool_stride)(conv)

    return conv, pool
#%%#####################################################################################################################
def bottle_neck(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal"
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    return conv
#%%#####################################################################################################################
def up_block(    
    input_tensor,
    no_filters,
    skip_connection, 
    kernel_size=(3, 3),
    strides=(1, 1),
    upsampling_factor = (2,2),
    max_pool_window = (2,2),
    padding="same",
    kernel_initializer="he_normal"):
    
    
    conv = Conv2D(
        filters = no_filters,
        kernel_size= max_pool_window,
        strides = strides,
        activation = None,
        padding = padding,
        kernel_initializer=kernel_initializer
    )(UpSampling2D(size = upsampling_factor)(input_tensor))
    
    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv) 
    
    
    conv = concatenate( [skip_connection , conv]  , axis = -1)
    
    
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)
    
    return conv
#%%#####################################################################################################################
def output_block(input_tensor,
    padding="same",
    kernel_initializer="he_normal"
):
    
    conv = Conv2D(
        filters=2,
        kernel_size=(3,3),
        strides=(1,1),
        activation="relu",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)
    
    
    conv = Conv2D(
        filters=1,
        kernel_size=(1,1),
        strides=(1,1),
        activation="sigmoid",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)
    
    
    return conv
#%%#####################################################################################################################
def UNet(input_shape = (128,128,3)):
    
    filter_size = [64,128,256,512,1024]
    
    inputs = Input(shape = input_shape)
    
    d1 , p1 = down_block(input_tensor= inputs,
                         no_filters=filter_size[0],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    d2 , p2 = down_block(input_tensor= p1,
                         no_filters=filter_size[1],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d3 , p3 = down_block(input_tensor= p2,
                         no_filters=filter_size[2],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d4 , p4 = down_block(input_tensor= p3,
                         no_filters=filter_size[3],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    b = bottle_neck(input_tensor= p4,
                         no_filters=filter_size[4],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal")
    
    
    
    u4 = up_block(input_tensor = b,
                  no_filters = filter_size[3],
                  skip_connection = d4,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    u3 = up_block(input_tensor = u4,
                  no_filters = filter_size[2],
                  skip_connection = d3,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u2 = up_block(input_tensor = u3,
                  no_filters = filter_size[1],
                  skip_connection = d2,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u1 = up_block(input_tensor = u2,
                  no_filters = filter_size[0],
                  skip_connection = d1,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    
    output = output_block(input_tensor=u1 , 
                         padding = "same",
                         kernel_initializer= "he_normal")
    
    model = keras.models.Model(inputs = inputs , outputs = output)
    
    
    return model
#%%#####################################################################################################################
# Set the optimizer for the model. In this case, the Adam optimizer with a learning rate of 1e-4 is used
model = UNet(input_shape = (128,128,3))
model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#%%#####################################################################################################################
# Also, we can plot the model by using the below code
from keras.utils import plot_model

# Visualize the model
plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)
#%%#####################################################################################################################
# Now, we train the model

images_path = f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/val"
masks_path = f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/val_mask"
batch_size = 8

val_generator = CustomDataGenerator(images_path, masks_path, batch_size)
#%%#####################################################################################################################
def print_preprocessed_image_shapes(model, generator):
    """
    Print the shapes of preprocessed images generated by the provided model and generator.

    Args:
        model (tf.keras.Model): The trained model.
        generator (CustomDataGenerator): Instance of the CustomDataGenerator class.
    """
    for i in range(len(generator)):
        # Get a batch of preprocessed images from the generator
        batch_images, batch_mask = generator[i]

        # Pass the batch of images through the model to obtain predictions
        # predictions = model.predict(batch_images)

        # Print the shapes of the preprocessed images
        for image in batch_images:
            print(f"Shape of preprocessed image: {image.shape}")
            
# Print the shapes of preprocessed images
print_preprocessed_image_shapes(model, val_generator)
#%%#####################################################################################################################
# Fit the model with the training generator
from tensorflow.python.keras.callbacks import TensorBoard
# TensorBoard logok elérési útja
log_dir = "./logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Tanítás TensorBoard logokkal
with tf.device('/GPU:0'):
    train_steps =  int(len(os.listdir( f"{user_path.replace(os.sep, '/')}/Downloads/ftw-dataset/train_mask"))/batch_size)
    model.fit(
        train_generator,
        validation_data = val_generator,
        steps_per_epoch = train_steps ,
        epochs=20,
        callbacks=[tensorboard_callback])
#%%#####################################################################################################################
# Get a sample batch from the validation data generator
sample_images, sample_masks = val_generator[0]

# Generate predictions on the sample batch
predictions = model.predict(sample_images)

# Threshold the predictions (if needed)
threshold = 0.5  # Adjust the threshold as per your requirement
thresholded_predictions = (predictions > threshold).astype(np.uint8)
#%%#####################################################################################################################
# Select a random index from the batch
idx = np.random.randint(0, sample_images.shape[0])

# Plot the sample image, ground truth mask, and predicted mask
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot sample image
sample_images_instance = sample_images[idx]
sample_images_instance = sample_images_instance[..., :3]
sample_images_instance = (sample_images_instance / np.max(sample_images_instance) * 255).astype(np.uint8)
axes[0].imshow(sample_images_instance)
axes[0].set_title('Sample Image')

# Plot ground truth mask
axes[1].imshow(sample_masks[idx])
axes[1].set_title('Ground Truth Mask')

# Plot predicted mask
axes[2].imshow(thresholded_predictions[idx])
axes[2].set_title('Predicted Mask')

# Set common title for the figure
fig.suptitle('Sample Image, Ground Truth Mask, and Predicted Mask')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()
# %%
