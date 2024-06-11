import numpy as np
import tensorflow as tf
import funcs
from sklearn.decomposition import PCA

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to a range of 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Number of images:")
print(" X Training data:", x_train.shape[0])
print(" X Test data:", x_test.shape[0])
total_images = x_train.shape[0] + x_test.shape[0]
print(" Total number of images = ", total_images)
print()

total_train_images = funcs.count_labels(y_train)
total_test_images = funcs.count_labels(y_test)

for value in total_train_images.keys():
    print(f"value: {value} occurs {total_test_images[value] + total_train_images[value]} times")
print()

train_white_pixels = funcs.count_white_pixels(x_train, y_train)
test_white_pixels = funcs.count_white_pixels(x_test, y_test)

total_white_pixels = {label: train_white_pixels[label] + test_white_pixels[label] for label in
                      train_white_pixels.keys()}
average_white_pixels, std_dev_white_pixels = funcs.calculate_statistics(total_white_pixels)

avg_white_pixels_dict = {}

for label in total_white_pixels.keys():
    avg_white_pixels = total_white_pixels[label] / (total_test_images[label] + total_train_images[label])
    avg_white_pixels_dict[label] = avg_white_pixels
    print(f"Average number of white pixels for Class {label}: {avg_white_pixels}")
print()

white_pixels_std = np.std(np.array(list(avg_white_pixels_dict.values())))
print(f"Standard deviation of white pixels: {white_pixels_std}")
print()

train_non_white_pixels = funcs.count_non_white_pixels(x_train, y_train)
test_non_white_pixels = funcs.count_non_white_pixels(x_test, y_test)

total_non_white_pixels = {label: train_non_white_pixels[label] + test_non_white_pixels[label] for label in train_white_pixels.keys()}

for label in total_non_white_pixels.keys():
    print(f"Total number of non-white pixels for Class {label}: {total_non_white_pixels[label]}")
print()

common_pixel_counts = {i: np.sum(total_non_white_pixels[i]) for i in range(10)}

print("Number of common non-white pixels per class:")
for label, count in common_pixel_counts.items():
    print(f"Class {label}: {count}")

epochs = [5, 10, 15]

# Build a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit class)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

histories = {}

# Train the model for 5 epochs
print()
print("Train for 5 epochs:")
histories[epochs[0]] = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=2)
# Train the model for 10 epochs
print()
print("Train for 10 epochs:")
histories[epochs[1]] = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=2)
# Train the model for 15 epochs
print()
print("Train for 15 epochs:")
histories[epochs[2]] = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=2)

y_preds = np.argmax(model.predict(x_test), axis=1)

funcs.print_scores(y_test, y_preds)
funcs.plot_conf_matrix(y_test, y_preds, total_white_pixels.keys(), 'Base Model')
funcs.plot_loss_to_epochs(histories, 'Base Model')

funcs.plot_img(x_train, 'Base Model')

x_train_avged = funcs.replace_pixel_by_average(x_train)
x_test_avged = funcs.replace_pixel_by_average(x_test)

second_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit class)
])

second_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

histories = {}

print()
print("Train new model with pixel value changed for 5 epochs:")
histories[epochs[0]] = second_model.fit(x_train_avged, y_train, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 10 epochs:")
histories[epochs[1]] = second_model.fit(x_train_avged, y_train, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 15 epochs:")
histories[epochs[2]] = second_model.fit(x_train_avged, y_train, epochs=5, validation_split=0.2, verbose=2)

y_preds = np.argmax(second_model.predict(x_test_avged), axis=1)

funcs.print_scores(y_test, y_preds)
funcs.plot_conf_matrix(y_test, y_preds, total_white_pixels.keys(), 'Averaged Pixels Model')
funcs.plot_loss_to_epochs(histories, 'Averaged Pixels Model')

funcs.plot_img(x_train_avged, 'Averaged Pixels Model')

# Flatten the images for PCA
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Perform PCA
pca = PCA(n_components=50)
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

# Restore the reduced dimensionality images back to their original shape
x_train_pca_img = pca.inverse_transform(x_train_pca).reshape(x_train.shape[0], 28, 28)
x_test_pca_img = pca.inverse_transform(x_test_pca).reshape(x_test.shape[0], 28, 28)

# Reshape data to fit the model
x_train_pca_img = x_train_pca_img.reshape(x_train_pca_img.shape[0], 28, 28, 1)
x_test_pca_img = x_test_pca_img.reshape(x_test_pca_img.shape[0], 28, 28, 1)

pca_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit class)
])

pca_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

histories = {}

print()
print("Train PCA model with pixel value changed for 5 epochs:")
histories[epochs[0]] = pca_model.fit(x_train_pca_img, y_train, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train PCA model with pixel value changed for 10 epochs:")
histories[epochs[1]] = pca_model.fit(x_train_pca_img, y_train, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train PCA model with pixel value changed for 15 epochs:")
histories[epochs[2]] = pca_model.fit(x_train_pca_img, y_train, epochs=5, validation_split=0.2, verbose=2)

y_preds = np.argmax(pca_model.predict(x_test_pca_img), axis=1)

funcs.print_scores(y_test, y_preds)
funcs.plot_conf_matrix(y_test, y_preds, total_white_pixels.keys(), "PCA Model")
funcs.plot_loss_to_epochs(histories, "PCA Model")

funcs.plot_img(x_train_pca_img, 'PCA Model')

x_train_non_overlap = funcs.block_averaging(x_train)
x_test_non_overlap = funcs.block_averaging(x_test)

x_train_non_overlap = x_train_non_overlap.reshape(x_train_non_overlap.shape[0], 28, 28, 1)
x_test_non_overlap = x_test_non_overlap.reshape(x_test_non_overlap.shape[0], 28, 28, 1)

non_overlapping_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit class)
])

non_overlapping_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

histories = {}

print()
print("Train new model with pixel value changed for 5 epochs:")
histories[epochs[0]] = non_overlapping_model.fit(x_train_avged, y_train, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 10 epochs:")
histories[epochs[1]] = non_overlapping_model.fit(x_train_avged, y_train, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 15 epochs:")
histories[epochs[2]] = non_overlapping_model.fit(x_train_avged, y_train, epochs=5, validation_split=0.2, verbose=2)

y_preds = np.argmax(non_overlapping_model.predict(x_test_non_overlap), axis=1)

funcs.print_scores(y_test, y_preds)
funcs.plot_conf_matrix(y_test, y_preds, total_white_pixels.keys(), 'Block Averaging Model')
funcs.plot_loss_to_epochs(histories, 'Block Averaging Model')

funcs.plot_img(x_test_non_overlap, 'Block Averaging Model')

x_train_4 = x_train[y_train == 4]
y_train_4 = y_train[y_train == 4]

x_test_4 = x_test[y_test == 4]
y_test_4 = y_test[y_test == 4]

x_train_8 = x_train[y_train == 8]
y_train_8 = y_train[y_train == 8]

x_test_8 = x_test[y_test == 8]
y_test_8 = y_test[y_test == 8]


print(f'Shape of x_train_4: {x_train_4.shape}')
print(f'Shape of y_train_4: {y_train_4.shape}')
print(f'Shape of x_train_8: {x_train_8.shape}')
print(f'Shape of y_train_8: {y_train_8.shape}')
print(f'Shape of x_test_4: {x_test_4.shape}')
print(f'Shape of y_test_4: {y_test_4.shape}')
print(f'Shape of x_test_8: {x_test_8.shape}')
print(f'Shape of y_test_8: {y_test_8.shape}')

new_size = (10, 10)

# Undersample the image
x_train_4_and_8 = np.concatenate((x_train_4, x_train_8), axis=0)
x_test_4_and_8 = np.concatenate((x_test_4, x_test_8), axis=0)

y_train_4_and_8 = np.concatenate((y_train_4, y_train_8), axis=0)
y_test_4_and_8 = np.concatenate((y_test_4, y_test_8), axis=0)

undersampled_images_train = funcs.undersample_images(x_train_4_and_8, new_size)
undersampled_images_test = funcs.undersample_images(x_test_4_and_8, new_size)

undersample_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit class)
])

undersample_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

histories = {}

print()
print("Train new model with pixel value changed for 5 epochs:")
histories[epochs[0]] = undersample_model.fit(undersampled_images_train, y_train_4_and_8, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 10 epochs:")
histories[epochs[1]] = undersample_model.fit(undersampled_images_train, y_train_4_and_8, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 15 epochs:")
histories[epochs[2]] = undersample_model.fit(undersampled_images_train, y_train_4_and_8, epochs=5, validation_split=0.2, verbose=2)

y_preds = np.argmax(undersample_model.predict(undersampled_images_test), axis=1)

funcs.print_scores(y_test_4_and_8, y_preds)
funcs.plot_conf_matrix(y_test_4_and_8, y_preds, {4, 8}, 'Undersampled Model')
funcs.plot_loss_to_epochs(histories, 'Undersampled Model')

funcs.plot_img(undersampled_images_train, 'Undersampled Model')


x_train_4_and_8_rotated = funcs.rotate_images(x_train_4_and_8, 45)
x_test_4_and_8_rotated = funcs.rotate_images(x_test_4_and_8, 45)

x_train_rotated_and_reg = np.concatenate((x_train_4_and_8, x_train_4_and_8_rotated), axis=0)
x_test_rotated_and_reg = np.concatenate((x_test_4_and_8, x_test_4_and_8_rotated), axis=0)

y_train_plus_rotated = np.concatenate((y_train_4_and_8, y_train_4_and_8), axis=0)
y_test_plus_rotated = np.concatenate((y_test_4_and_8, y_test_4_and_8), axis=0)

rotated_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit class)
])

rotated_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

histories = {}

print()
print("Train new model with pixel value changed for 5 epochs:")
histories[epochs[0]] = rotated_model.fit(x_train_rotated_and_reg, y_train_plus_rotated, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 10 epochs:")
histories[epochs[1]] = rotated_model.fit(x_train_rotated_and_reg, y_train_plus_rotated, epochs=5, validation_split=0.2, verbose=2)
print()
print("Train new model with pixel value changed for 15 epochs:")
histories[epochs[2]] = rotated_model.fit(x_train_rotated_and_reg, y_train_plus_rotated, epochs=5, validation_split=0.2, verbose=2)

y_preds = np.argmax(rotated_model.predict(x_test_rotated_and_reg), axis=1)

funcs.print_scores(y_test_plus_rotated, y_preds)
funcs.plot_conf_matrix(y_test_plus_rotated, y_preds, {4, 8}, 'Rotated Model')
funcs.plot_loss_to_epochs(histories, 'Rotated Model')

funcs.plot_rotated_image(x_train_4_and_8[0], x_train_4_and_8_rotated[0])


