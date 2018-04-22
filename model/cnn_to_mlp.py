from tensorflow.contrib import layers
import tensorflow as tf


def _cnn_to_mpl(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out=inpt
        out_shape = out.get_shape().as_list()
        print("out_shape: ", out_shape)
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
                out_shape = out.get_shape().as_list()
                print("out_shape: ", out_shape)
        conv_out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out)
                    state_out = tf.nn.relu(state_out)
                state_value = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_score_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_score_mean, 1)
                q_out = state_value + action_scores_centered
        else:
            q_out = action_scores
        return q_out

def cnn_to_mlp(convs, hiddens, duelings=False, layer_norm=False):
    return lambda *args, **kwargs: _cnn_to_mpl(convs, hiddens, duelings, *args, **kwargs)