import tensorflow as tf
from tensorflow.keras import layers, models
import wandb
from wandb.keras import WandbCallback
from huggingface_hub import login, HfApi

# Initialize Weights & Biases
wandb.init(project="trash-classification", entity="your-username")

# Define dataset path
data_dir = "thrashnet_dataset"  # Replace with your dataset directory
batch_size = 32
img_height = 224
img_width = 224

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Define class names
class_names = train_ds.class_names
print(f"Class Names: {class_names}")

# Define CNN model
model = models.Sequential
