#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, w3, w4, w7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # 1) Implement 1x1 convs on top of vgg layer 7
    # 2) Implement the decoder part from the semantic segmentation paper (and classromm)
    # 3) Return the output tensor
    # 4) Profit

    # Using keyword argument packing to reduce duplicate code and make modifications quicker.
    common_params = {"filters": num_classes,
                     "padding": "same",
                     "kernel_regularizer": tf.contrib.layers.l2_regularizer(1e-5),
                     "kernel_initializer": tf.initializers.truncated_normal(mean=0.0, stddev=1e-5)}

    params_1x1 = {"kernel_size": 1, "strides": (1, 1), **common_params}
    params_2x = {"kernel_size": 4, "strides": (2, 2), **common_params}
    params_8x = {"kernel_size": 16, "strides": (8, 8), **common_params}

    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, **params_1x1)
    layer7_2x = tf.layers.conv2d_transpose(layer7_conv_1x1, **params_2x)

    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, **params_1x1)
    layer_2x_combo = tf.add(layer7_2x, layer4_conv_1x1)

    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, **params_1x1)
    layer_4x = tf.layers.conv2d_transpose(layer_2x_combo, **params_2x)
    layer_4x_combo = tf.add(layer_4x, layer3_conv_1x1)

    return tf.layers.conv2d_transpose(layer_4x_combo, **params_8x)

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    tvars = tf.trainable_variables()
    print("tvars: ")
    print(tvars)
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            print("STARTING NEXT BATCH")
            print("Label original shape:")
            print(label.shape)
            try:
                labels = tf.reshape(label, (-1, 2))
                print("Labels shape: %s" % labels.get_shape())
            except ValueError:
                pass

            out, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 1.0, learning_rate: 0.5})


            ## Get rid of this mess eventually
            # try:
            #     print(out.get_shape())
            # except AttributeError:
            #     print("no shape for out:")
            #     print(out)
            # try:
            #     print("Printing output shape")
            #     outs = tf.reshape(out, (-1, 2))
            #     print("Output shape: %s" % outs.get_shape())
            # except ValueError:
            #     print("Something happened! Noopoe")


            print("Loss: = {:.3f}".format(loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    num_epochs = 3
    batch_size = 128
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    correct_labels = tf.placeholder(dtype=tf.int8, shape=(None, image_shape[0], image_shape[1], num_classes))
    learning_rate_placeholder = tf.placeholder(dtype='float32')

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output_layer = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_labels,
                                                        learning_rate_placeholder, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_labels, keep_prob, learning_rate_placeholder)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        return


if __name__ == '__main__':
    run()
