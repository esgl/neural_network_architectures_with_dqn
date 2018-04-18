import tensorflow.contrib.layers as layers
import tensorflow as tf

def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out=inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def mpl(hiddens=[], layer_norm=False):
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


def _cnn_to_mpl(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out=inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
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

def _cnn_to_lstm(convs, lstm_cell_size, hiddens, batch_size, duelings, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

            # convs_out = layers.flatten(out)
            print("type(out):", type(out))
            print("out.shape", out.get_shape().as_list())
            print("shape[1]", out.get_shape().as_list()[3])
            print("shape[2]", out.get_shape().as_list()[1] * out.get_shape().as_list()[2])
            out_shape = out.get_shape().as_list()
            convs_out = tf.reshape(out, [-1, out_shape[3], out_shape[1] * out_shape[2]])
            print("type(convs_out):", type(convs_out))
            print("convs_out.shape", convs_out.get_shape().as_list())

        with tf.variable_scope("lstm"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_cell_size, forget_bias=1.0, state_is_tuple=True)
            with tf.variable_scope("initial_state"):
                cell_init_state = lstm_cell.zero_state(batch_size=out_shape[0],
                                                       dtype=tf.float32)
            cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=convs_out,
                initial_state=cell_init_state,
                time_major=True
            )

            out = cell_final_state
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
        print("q_out.shape", q_out.get_shape().as_list())
        return q_out

def cnn_to_lstm(convs, lstm_cell_size, hiddens, batch_size, dueling=False, layer_norm=False):
    return lambda *args, **kwargs: _cnn_to_lstm(convs,
                                                lstm_cell_size,
                                                hiddens,
                                                batch_size,
                                                dueling,
                                                layer_norm=layer_norm,
                                                *args, **kwargs)




