import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow_model_optimization as tfmot
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply          = tfmot.quantization.keras.quantize_apply

def softJaccardLoss(yTrue, yPred, epsilon=1e-9):
    yTrue = tf.cast(yTrue, tf.float32)
    yPred = tf.cast(yPred, tf.float32)

    # Flatten per-sample to compute pixel-wise over the whole image
    yTrueFlat = tf.reshape(yTrue, [tf.shape(yTrue)[0], -1])
    yPredFlat = tf.reshape(yPred, [tf.shape(yPred)[0], -1])

    intersection = tf.reduce_sum(yTrueFlat * yPredFlat, axis=1)
    sum_ = tf.reduce_sum(yTrueFlat + yPredFlat, axis=1)
    jaccard = (intersection + epsilon) / (sum_ - intersection + epsilon)

    return 1 - jaccard  # Still returns shape [B]; Keras averages automatically

def diceCoefficient(yTrue, yPred, smooth=1):
    yTrueF = tf.reshape(yTrue, [-1])
    yPredF = tf.reshape(yPred, [-1])
    intersection = tf.reduce_sum(yTrueF * yPredF)
    return (2. * intersection + smooth) / (tf.reduce_sum(yTrueF) + tf.reduce_sum(yPredF) + smooth)

def diceLoss(yTrue, yPred):
    return 1 - diceCoefficient(yTrue, yPred)


def quantConvBlock(x, filters, kernelSize=3, activation='relu', name=None):
    x = quantize_annotate_layer(
        Conv2D(filters, kernelSize, padding='same', name=f"{name}_conv1" if name else None)
    )(x)
    x = BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = Activation(activation, name=f"{name}_act1" if name else None)(x)

    x = quantize_annotate_layer(
        Conv2D(filters, kernelSize, padding='same', name=f"{name}_conv2" if name else None)
    )(x)
    x = BatchNormalization(name=f"{name}_bn2" if name else None)(x)
    x = Activation(activation, name=f"{name}_act2" if name else None)(x)
    return x

def uNetQ(batchShape, filters=32):
    inputs = Input(batch_shape=batchShape)

    # Encoder (6 blocks)
    c1 = quantConvBlock(inputs, filters, name="enc1")
    p1 = MaxPooling2D((2, 2), name="pool1")(c1)

    c2 = quantConvBlock(p1, filters*2, name="enc2")
    p2 = MaxPooling2D((2, 2), name="pool2")(c2)

    c3 = quantConvBlock(p2, filters*4, name="enc3")
    p3 = MaxPooling2D((2, 2), name="pool3")(c3)

    c4 = quantConvBlock(p3, filters*8, name="enc4")
    p4 = MaxPooling2D((2, 2), name="pool4")(c4)

    c5 = quantConvBlock(p4, filters*16, name="enc5")
    p5 = MaxPooling2D((2, 2), name="pool5")(c5)

    c6 = quantConvBlock(p5, filters*32, name="bottleneck")  # 6th (bottleneck)

    # Decoder (mirrored, 5 up blocks)
    u5 = quantize_annotate_layer(
        Conv2DTranspose(filters*16, (2, 2), strides=(2, 2), padding='same', name="up5")
    )(c6)
    u5 = Concatenate(name="concat5")([u5, c5])
    d5 = quantConvBlock(u5, filters*16, name="dec5")

    u4 = quantize_annotate_layer(
        Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same', name="up4")
    )(d5)
    u4 = Concatenate(name="concat4")([u4, c4])
    d4 = quantConvBlock(u4, filters*8, name="dec4")

    u3 = quantize_annotate_layer(
        Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same', name="up3")
    )(d4)
    u3 = Concatenate(name="concat3")([u3, c3])
    d3 = quantConvBlock(u3, filters*4, name="dec3")

    u2 = quantize_annotate_layer(
        Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same', name="up2")
    )(d3)
    u2 = Concatenate(name="concat2")([u2, c2])
    d2 = quantConvBlock(u2, filters*2, name="dec2")

    u1 = quantize_annotate_layer(
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name="up1")
    )(d2)
    u1 = Concatenate(name="concat1")([u1, c1])
    d1 = quantConvBlock(u1, filters, name="dec1")

    outputs = quantize_annotate_layer(
        Conv2D(1, (1, 1), activation='sigmoid', name="output")
    )(d1)

    model = Model(inputs, outputs)
    model = quantize_apply(model)
    model.compile(optimizer='adam', loss=softJaccardLoss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), diceCoefficient])

    return model

# Example usage
if __name__ == "__main__":
    model = uNetQ(batchShape=(4, 384, 384, 4))
