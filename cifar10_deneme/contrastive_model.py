import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import random

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_test = X_train/255, X_test/255

class_order = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create a list to store the ordered training items
ordered_train_images = []
ordered_train_labels = []

# Iterate over the class order and extract the corresponding training items
for i in range(len(class_order)):
    class_name = class_order[i]
    class_index = i
    class_indices = np.where(y_train == class_index)[0]
    class_images = X_train[class_indices]
    class_labels = y_train[class_indices]
    ordered_train_images.extend(class_images)
    ordered_train_labels.extend(class_labels)
# Convert the ordered training items back to numpy arrays
X_train = np.array(ordered_train_images)
y_train = np.array(ordered_train_labels)


X_train = np.concatenate([X_train[:1000], X_train[5000:6000],
                          X_train[10000:11000], X_train[15000:16000],
                          X_train[20000:21000], X_train[25000:26000],
                          X_train[30000:31000], X_train[35000:36000],
                          X_train[40000:41000], X_train[45000:46000]])
y_train = np.concatenate([y_train[:1000], y_train[5000:6000],
                          y_train[10000:11000], y_train[15000:16000],
                          y_train[20000:21000], y_train[25000:26000],
                          y_train[30000:31000], y_train[35000:36000],
                          y_train[40000:41000], y_train[45000:46000]])

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

style_images = [load_image("4.jpeg"), load_image("frida.jpg"), load_image("1.jpg")]


content_images = []

for i in range(10):
    content = tf.image.convert_image_dtype(X_train[i], tf.float32)
    content = content[tf.newaxis, :]
    content_images.append(content)


def generate_samples(content_images, style_images, style_transfer_model):
    positive_samples = []
    negative_samples = []

    # Generate positive samples
    for content in content_images:
        for style in style_images:
            stylized_image = style_transfer_model(tf.constant(content), tf.constant(style))[0]
            positive_samples.append((content, stylized_image))

    # Generate negative samples
    for content1 in content_images:
        for content2 in content_images:
            if not (np.array_equal(content1, content2)):
                random_style = style_images[np.random.randint(0, len(style_images) - 1)]
                stylized_image = style_transfer_model(tf.constant(content1), tf.constant(random_style))[0]
                negative_samples.append((content1, stylized_image))

    return positive_samples, negative_samples




# Define the contrastive model architecture
def create_contrastive_model(input_shape, embedding_dim):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(embedding_dim, activation='relu')(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

positive_samples, negative_samples = generate_samples(content_images, style_images, model)
all_samples = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)


combined = list(zip(all_samples, labels))
random.shuffle(combined)

all_samples, labels = zip(*combined)

all_samples = tf.convert_to_tensor(all_samples)
labels = tf.convert_to_tensor(labels)


train_size = int(0.8 * len(all_samples))
train_samples, val_samples = all_samples[:train_size], all_samples[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]


# Reshape the samples to remove the extra dimension
train_samples = tf.reshape(train_samples, (-1, ) + train_samples.shape[3:])
val_samples = tf.reshape(val_samples, (-1, ) + val_samples.shape[3:])

# Create the contrastive model
input_shape = (32, 32, 3)  # Modify according to your image shape
embedding_dim = 128  # Modify according to your desired embedding dimension
contrastive_model = create_contrastive_model(input_shape, embedding_dim)

# Define the contrastive loss function
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.square(1 - y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Compile the model
contrastive_model.compile(optimizer='adam', loss=contrastive_loss)

# Train the contrastive model
contrastive_model.fit(train_samples, train_labels, validation_data=(val_samples, val_labels), epochs=10)












