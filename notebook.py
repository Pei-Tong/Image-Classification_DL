#!/usr/bin/env python
# coding: utf-8

# In[27]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from collections import Counter

# Spilt dataset (Base on layer)
import os #Iterate folder, create new catalog
import shutil #File operation - move file, etc
from sklearn.model_selection import train_test_split

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Pre-training Model
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout


# ### Data Preparation ans data cleaning <br>
# Dataset: Intel Image Classification<br>
# https://www.kaggle.com/datasets/puneet6060/intel-image-classification <br>
# - Load dataset<br>
# - spilt to 3 dataset - train/validation/test<br>
# - Image Normalization and standardlization<br>

# In[28]:


train_dir = './intel-image-classification/seg_train/seg_train'
test_dir = './intel-image-classification/seg_test/seg_test'


# In[29]:


IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32


# #### EDA  (Exploratory Data Analysis)
# - Data structure
# - Img/Label shape
# - Class distribution
# - Visualize pictures

# In[30]:


# Load dataset, Implicit regularization
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# In[31]:


# View dataset information
for images, labels in train_ds.take(1): #Extract one batch
    print(f"Images shape: {images.shape}")  # fig shape
    print(f"Labels shape: {labels.shape}")  # label shape
    print(f"Image max value: {tf.reduce_max(images)}")
    print(f"Image min value: {tf.reduce_min(images)}")  

for images, labels in test_ds.take(1):    
    print(f"Images shape: {images.shape}")  # fig shape
    print(f"Labels shape: {labels.shape}")  # label shape


# In[32]:


class_list = train_ds.class_names
print(f"class_names: {class_list}")


# In[33]:


label_count = Counter()

for image, labels in train_ds: #batch
    label_indices = labels.numpy()
    for index in label_indices:
        class_names = class_list[index]
        label_count[class_names] += 1
for class_name, count in sorted(label_count.items()):
    print(f"{class_name}: {count}")


# In[34]:


# Check label distribution
classes, counts = zip(*label_count.items())

plt.figure(figsize=(5,4))
plt.bar(classes, counts, color='green')
plt.title('Class Distribution in Training Dataset')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[35]:


# Visible data
plt.figure(figsize=(5, 5))
for images, labels in train_ds.take(1):  # Extract one batch
    for i in range(9):  # Display first 9 pics
        ax = plt.subplot(3, 3, i + 1) #layout
        plt.imshow(images[i].numpy().astype("uint8"))  # img convert to NumPy and display
        label_index = labels[i].numpy()  # Extract the numeric index of labels
        plt.title(class_list[label_index])  # Map num index to a text class
        plt.axis("off")
plt.tight_layout()
plt.show()


# #### Spilt Dataset

# In[36]:


val_dir = './intel-image-classification/seg_val'
os.makedirs(val_dir, exist_ok=True) #No error when directory already exists


# In[37]:


#Go through every folder - category
for class_name in class_list:
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path): #make sure it's a folder
        continue
    
    images = os.listdir(class_path)
    train_images, val_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )
    
    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True) #No error when directory already exists
    
    for img_name in val_images:
        src_path = os.path.join(class_path, img_name)
        dest_path = os.path.join(val_class_path, img_name)
        shutil.move(src_path, dest_path) # move from source to target path
    


# In[38]:


#Go through every folder - category
for class_name in class_list:
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path): #make sure it's a folder
        continue
    
    images = os.listdir(class_path)
    train_images, val_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )
    
    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True) #No error when directory already exists
    
    for img_name in val_images:
        src_path = os.path.join(class_path, img_name)
        dest_path = os.path.join(val_class_path, img_name)
        shutil.move(src_path, dest_path) # move from source to target path
    


# In[39]:


def count_images_in_dir(directory):
    
    total_count = 0
    print(f"{directory}")
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"{class_name}: {num_images} pictures")
            total_count += num_images
    print(f"Total Pictures: {total_count}")
label_count = Counter()

count_images_in_dir('./intel-image-classification/seg_train/seg_train')
count_images_in_dir('./intel-image-classification/seg_val')


# ### Define Model

# In[40]:


train_dir = './intel-image-classification/seg_train/seg_train'
test_dir = './intel-image-classification/seg_test/seg_test'
val_dir = './intel-image-classification/seg_val'


# #### Transfer learning

# In[41]:


# Image augmentation and preprocessing
# Pre-process, scale px value to [-1.1]
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest')
train_ds = train_gen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='sparse'
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_ds = test_gen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='sparse'
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    val_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='sparse'
)


# In[42]:


images, labels = next(train_ds)  # 提取一個批次
print(f"Images shape: {images.shape}")  # 圖片形狀
print(f"Labels shape: {labels.shape}")  # 標籤形狀
print(f"Pixel range: {images.min()} to {images.max()}")


# #### Base Model Structure

# In[43]:


# Define Model
def create_model(
    input_shape = (150,150,3),
    base_model_name = 'Xception',
    trainable=False,
    learning_rate = 0.001,
    size_inner = 128,
    droprate=0.5,
    num_classes = len(train_ds.class_indices),
):

    if base_model_name == 'Xception':
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    base_model.trainable = trainable #Freeze or unfreeze the parameters of the pre-training model

    #model structure
    inputs = Input(shape=input_shape) #input layer
    base = base_model(inputs, training=False) #pre-training extract features
    vectors= GlobalAveragePooling2D()(base) #pooling, 4D to 2D
    dropout_vectors = Dropout(droprate)(vectors) #dropout: random abandon some features
    inner = Dense(size_inner, activation = 'relu')(dropout_vectors) #relu activation
    dropout_inner = Dropout(droprate)(inner) #dropout: enhanced normalization
    outputs = Dense(num_classes, activation='softmax')(dropout_inner)

    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model


# ### Model selection process and parameter tuning

# In[45]:


# Learning Rate
histories_lr = {}

for lr in [0.0001, 0.001, 0.01]:
    print(f"\nTraining model with learning_rate={lr}")
    
    model = create_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=5, validation_data=val_ds)
    
    histories_lr[f"lr={lr}"] = history.history


# In[46]:


plt.figure(figsize=(12, 5))
for lr, hist in histories_lr.items():
    plt.plot(hist["accuracy"], label=f"Train {lr}")
    plt.plot(hist["val_accuracy"], '--', label=f"Validation {lr}")

plt.title("Accuracy Comparison Across Learning Rates")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
for lr, hist in histories_lr.items():
    plt.plot(hist["loss"], label=f"Train {lr}")
    plt.plot(hist["val_loss"], '--', label=f"Validation {lr}")

plt.title("Loss Comparison Across Learning Rates")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[47]:


# Model selection
histories_model = {}

for model_name in ['Xception', 'ResNet50']:
    print(f"\nTraining model with model = {model_name}")
    
    model = create_model(base_model_name = model_name)
    history = model.fit(train_ds, epochs=5, validation_data=val_ds)
    
    histories_model[f"model={model_name}"] = history.history


# In[48]:


plt.figure(figsize=(12, 5))
for model_name, hist in histories_model.items():
    plt.plot(hist["accuracy"], label=f"Train {model_name}")
    plt.plot(hist["val_accuracy"], '--', label=f"Validation {model_name}")

plt.title("Accuracy Comparison Across Model Name")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
for model_name, hist in histories_model.items():
    plt.plot(hist["loss"], label=f"Train {model_name}")
    plt.plot(hist["val_loss"], '--', label=f"Validation {model_name}")

plt.title("Loss Comparison Across Across Model Name")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[49]:


# Size inner
histories_size_inner = {}

for size_inner in [64, 128, 256]:
    print(f"\nTraining model with size_inner = {size_inner}")
    
    model = create_model(size_inner = size_inner)
    history = model.fit(train_ds, epochs=5, validation_data=val_ds)
    
    histories_size_inner[f"size_inner={size_inner}"] = history.history


# In[50]:


plt.figure(figsize=(12, 5))
for size_inner, hist in histories_size_inner.items():
    plt.plot(hist["accuracy"], label=f"Train {size_inner}")
    plt.plot(hist["val_accuracy"], '--', label=f"Validation {size_inner}")

plt.title("Accuracy Comparison Across Size Inner")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
for size_inner, hist in histories_size_inner.items():
    plt.plot(hist["loss"], label=f"Train {size_inner}")
    plt.plot(hist["val_loss"], '--', label=f"Validation {size_inner}")

plt.title("Loss Comparison Across Across Size Inner")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[51]:


# Droprate
histories_droprate = {}

for droprate in [0.3, 0.5, 0.7]:
    print(f"\nTraining model with droprate = {droprate}")
    
    model = create_model(droprate = droprate)
    history = model.fit(train_ds, epochs=5, validation_data=val_ds)
    
    histories_droprate[f"droprate={droprate}"] = history.history


# In[52]:


plt.figure(figsize=(12, 5))
for droprate, hist in histories_droprate.items():
    plt.plot(hist["accuracy"], label=f"Train {droprate}")
    plt.plot(hist["val_accuracy"], '--', label=f"Validation {droprate}")

plt.title("Accuracy Comparison Across Droprate")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
for droprate, hist in histories_droprate.items():
    plt.plot(hist["loss"], label=f"Train {droprate}")
    plt.plot(hist["val_loss"], '--', label=f"Validation {droprate}")

plt.title("Loss Comparison Across Across Droprate")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

