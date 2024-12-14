import tensorflow as tf
from tensorflow.keras import layers

def build_model(image_size): # Note: explanations provided as part of final report
    input_layer = tf.keras.Input(shape=(image_size[0], image_size[1], 1))

    # CNN 
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # transformer Layers
    transformer_input = layers.Reshape((1, -1))(x)  # reshape CNN output for Transformer compatibility
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(transformer_input, transformer_input)
    transformer_features = layers.GlobalAveragePooling1D()(attention_output)

    # concatenate CNN and transformer features
    combined_features = layers.Concatenate()([x, transformer_features])
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined_features)
    x = layers.Dropout(0.4)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model
