from mne.time_frequency import psd_array_welch
import pickle as pk
from keras import layers, models, Input
from keras.src.models import Model
import pickle
from keras.layers import Dense
import tensorflow as tf
import keras
import numpy as np

with open("my_eeg_epoched_data_30_inf.pkl", "rb") as f:
    epoched_data = pickle.load(f)

x1_train = epoched_data["x1_train"]
x2_train = epoched_data["x2_train"]
y_train = epoched_data["y_train"]

x1_val = epoched_data["x1_val"]
x2_val = epoched_data["x2_val"]
y_val = epoched_data["y_val"]

x1_test = epoched_data["x1_test"]
x2_test = epoched_data["x2_test"]
y_test = epoched_data["y_test"]

index_list_test = epoched_data["index_list_test"]
valid_channels_name = epoched_data["valid_channels_name"]

valid_channels_length = len(valid_channels_name)

def build_eegnet():
    input_layer = Input(shape=(valid_channels_length, 625, 1))

    # Temporal Convolution
    x = layers.Conv2D(8, (1, valid_channels_length), padding='same', activation=None, name="conv_temporal")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)

    # Depthwise Convolution (Spatial Filtering)
    x = layers.DepthwiseConv2D((valid_channels_length, 1), use_bias=False, depth_multiplier=2, padding='valid', name="conv_spatial")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(0.5)(x)

    # Separable Convolution (Extracting Temporal-Spatial Features)
    x = layers.SeparableConv2D(16, (1, 16), padding='same', activation=None, name="conv_separable")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(0.5)(x)

    # Flatten and Dense Embedding
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    return Model(inputs=input_layer, outputs=x)

cnn = build_eegnet()

input_1 = Input(shape=(len(valid_channels_name), 625, 1))
input_2 = Input(shape=(len(valid_channels_name), 625, 1))

# Shared feature extractor
encoded_1 = cnn(input_1)
encoded_2 = cnn(input_2)

# Ensure output shape before distance calculation
print("Encoded_1 shape:", encoded_1.shape)
print("Encoded_2 shape:", encoded_2.shape)

# L1 Distance for Temporal Comparison
# Ensure we calculate the absolute difference between two feature vectors of shape (None, 16)
@keras.saving.register_keras_serializable()
class L1DistanceLayer(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return tf.math.abs(x1 - x2)

# Use this in your model instead of Lambda

distance = L1DistanceLayer()([encoded_1, encoded_2])

# Classification Layer: Output is binary (sigmoid for binary classification)
output = Dense(1, activation='sigmoid')(distance)

# Compile Model
siamese_model = Model(inputs=[input_1, input_2], outputs=output)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

siamese_model.summary()

# Print final dataset shapes
print(f"Train set: x1_train {x1_train.shape}, x2_train {x2_train.shape}, y_train {y_train.shape}")
print(f"Validation set: x1_val {x1_val.shape}, x2_val {x2_val.shape}, y_val {y_val.shape}")
print(f"Test set: x1_test {x1_test.shape}, x2_test {x2_test.shape}, y_test {y_test.shape}")
# Training
batch_size = 16
siamese_model.fit([np.array(x1_train), np.array(x2_train)], np.array(y_train), epochs=60, batch_size=batch_size,
                  validation_data=([x1_val, x2_val], y_val))

test_loss, test_acc = siamese_model.evaluate([np.array(x1_test), np.array(x2_test)], np.array(y_test))

print(f"Test Accuracy: {test_acc:.4f}")

siamese_model.save("saved_models/siamese_model_61ch_60epochs_30_inf.keras")  # Saves in HDF5 format