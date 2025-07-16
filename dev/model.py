import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow_model_optimization as tfmot
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply          = tfmot.quantization.keras.quantize_apply

def softJaccardLoss(yTrue, yPred, epsilon=1e-6):
    yTrue = tf.cast(yTrue, tf.float32)
    yPred = tf.cast(yPred, tf.float32)

    # Flatten per-sample to compute pixel-wise over the whole image
    yTrueFlat = tf.reshape(yTrue, [tf.shape(yTrue)[0], -1])
    yPredFlat = tf.reshape(yPred, [tf.shape(yPred)[0], -1])

    intersection = tf.reduce_sum(yTrueFlat * yPredFlat, axis=1)
    sum_ = tf.reduce_sum(yTrueFlat + yPredFlat, axis=1)
    jaccard = (intersection + epsilon) / (sum_ - intersection + epsilon)

    return 1 - jaccard  # Still returns shape [B]; Keras averages automatically


def uNetq(batchShape):
    
    inputs = Input(batch_shape=batchShape, name="input_1")

    #encoder
    conv1 = quantize_annotate_layer(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))(inputs)
    pool1 = MaxPooling2D(pool_size=2, padding='same')(conv1)
    #bottleneck
    bottleneck = quantize_annotate_layer(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))(pool1)
    #decoder
    up1 = quantize_annotate_layer(Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'))(bottleneck)
    concat1 = Concatenate()([up1, conv1])
    conv3 = quantize_annotate_layer(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))(concat1)

    outputs = quantize_annotate_layer(Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same'))(conv3)

    model = Model(inputs=inputs, outputs=outputs)
    quantizedModel = quantize_apply(model)
    quantizedModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    quantizedModel.summary()

    plot_model(model, to_file="dev/uNet_model.png", show_shapes=True, show_layer_names=True)

    return quantizedModel


if __name__ == "__main__":
    uNetq(batchShape=(4,384,384,4))
