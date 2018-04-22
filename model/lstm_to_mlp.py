import tensorflow as tf
from tensorflow.contrib import layers

def _lstm_to_mlp(lstm_hidden_size, lstm_out_size, hiddens, batch_size, duelings,
                 inpt, n_actions, scope, reuse=False, layer_norm=False):
    print(reuse)
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        print("out.shape:{}".format(out.get_shape().as_list()))
        out_shape = out.get_shape().as_list()
        n_steps = out_shape[3]
        n_inputs = out_shape[1] * out_shape[2]
        print("n_step:{}, n_inputs:{}".format(n_steps, n_inputs))

        with tf.variable_scope("weights"):
            weights = {
                "in" : tf.Variable(tf.random_normal([n_inputs, lstm_hidden_size])),
                "out": tf.Variable(tf.random_normal([lstm_hidden_size, lstm_out_size]))
            }
        with tf.variable_scope("biases"):
            biases = {
                "in": tf.constant(0.1, shape=[lstm_hidden_size, ]),
                "out": tf.constant(0.1, shape=[lstm_out_size, ])
            }

        with tf.variable_scope("lstm"):
            out = tf.reshape(out, [-1, n_inputs])
            print("out2.shape:{}".format(out.get_shape().as_list()))
            out_lstm = tf.matmul(out, weights["in"]) + biases["in"]
            print("out_lstm1.shape:{}".format(out_lstm.get_shape().as_list()))
            out_lstm = tf.reshape(out_lstm, [-1, n_steps, lstm_hidden_size])
            print("out_lstm2.shape:{}".format(out_lstm.get_shape().as_list()))

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0, state_is_tuple=True,
                                                     name="basiclstmcell")
            init_state = lstm_cell.zero_state(batch_size , dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=out_lstm,
                initial_state=init_state,
                time_major=False
            )
            print("outputs.shape:{}".format(outputs.get_shape().as_list()))

            out = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

            print("out.shape:", out)
            out = out[-1]
            print("out3.shape:{}".format(out.get_shape().as_list()))
            print('weights["out"].shape:{}'.format(weights["out"].get_shape().as_list()))
            out = tf.matmul(out, weights["out"]) + biases["out"]
            print("out4.shape:{}".format(out.get_shape().as_list()))

        with tf.variable_scope("action_value"):
            action_out = tf.reshape(out, [-1, lstm_out_size])
            print("action_out.shape:{}".format(action_out.get_shape().as_list()))
            for hidden in hiddens:
                action_out = layers.fully_connected(inputs=action_out,
                                             num_outputs=hidden,
                                             activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(inputs=action_out,
                                                  num_outputs=n_actions,
                                                  activation_fn=None)
        if duelings:
            with tf.variable_scope("state_value"):
                state_out = tf.reshape(out, [-1, lstm_out_size])
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


def lstm_to_mlp(lstm_hidden_size, lstm_out_size, hiddens, batch_size, duelings=False, layer_norm=False):
    return lambda *args, **kwargs: _lstm_to_mlp(lstm_hidden_size,
                                                lstm_out_size,
                                                hiddens,
                                                batch_size,
                                                duelings,
                                                *args,
                                                **kwargs)