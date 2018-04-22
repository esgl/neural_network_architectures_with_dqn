import tensorflow as tf
from tensorflow.contrib import layers

def _cnn_to_lstm(convs, lstm_hidden_size, lstm_out_size, hiddens, batch_size, duelings, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)



        with tf.variable_scope("lstm"):
            out_shape = out.get_shape().as_list()
            n_steps = out_shape[3]
            n_inputs = out_shape[1] * out_shape[2]
            convs_out = tf.reshape(out, [-1, n_steps, n_inputs])

            print("convs_out.shape: {}".format(convs_out.get_shape().as_list()))
            with tf.variable_scope("weights"):
                weights = {
                    "in": tf.Variable(tf.random_normal([n_inputs, lstm_hidden_size])),
                    "out": tf.Variable(tf.random_normal([lstm_hidden_size, lstm_out_size]))
                }
            with tf.variable_scope("biases"):
                biases = {
                    "in": tf.constant(0.1, shape=[lstm_hidden_size, ]),
                    "out": tf.constant(0.1, shape=[lstm_out_size, ])
                }
            lstm_in = tf.reshape(convs_out, [-1, n_inputs])
            print("lstm_in.shape: {}".format(lstm_in.get_shape().as_list()))
            lstm_in = tf.matmul(lstm_in, weights["in"]) + biases["in"]
            lstm_in = tf.reshape(lstm_in, [-1, n_steps, lstm_hidden_size])
            print("lstm_in.shape: {}".format(lstm_in.get_shape().as_list()))

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0, state_is_tuple=True)
            cell_init_state = lstm_cell.zero_state(batch_size=batch_size,
                                                       dtype=tf.float32)
            cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=lstm_in,
                initial_state=cell_init_state,
                time_major=False
            )
            out = tf.unstack(tf.transpose(cell_outputs, [1, 0, 2]))
            out = tf.matmul(out[-1], weights["out"]) + biases["out"]

        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(inputs=action_out,
                                                    num_outputs=hidden,
                                                    activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(inputs=action_out,
                                                num_outputs=num_actions,
                                                activation_fn=None)
        if duelings:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(inputs=state_out,
                                                       num_outputs=hidden,
                                                       activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(inputs=state_out,
                                                   num_outputs=1,
                                                   activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out

def cnn_to_lstm(convs, lstm_hidden_size, lstm_out_size, hiddens, batch_size, duelings=False, layer_norm=False):
    return lambda *args, **kwargs: _cnn_to_lstm(convs,
                                                lstm_hidden_size,
                                                lstm_out_size,
                                                hiddens,
                                                batch_size,
                                                duelings,
                                                layer_norm=layer_norm,
                                                *args, **kwargs)