import numpy as np
import tensorflow as tf


class NST:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) or len(style_image.shape) != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or len(content_image.shape) != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not (isinstance(alpha, (int, float)) and alpha >= 0):
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, (int, float)) and beta >= 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, _ = image.shape
        max_dim = 512
        if h > w:
            h_new = max_dim
            w_new = int(w * (max_dim / h))
        else:
            w_new = max_dim
            h_new = int(h * (max_dim / w))

        image = tf.image.resize(np.expand_dims(image, axis=0), size=(h_new, w_new), method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

