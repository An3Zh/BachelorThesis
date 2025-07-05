import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

def simple(inputShape):
    inputs = tf.keras.Input(shape=inputShape)

    x = Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)

    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='relu', padding='same')(x)

    outputs = Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model