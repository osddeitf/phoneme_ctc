OLD_CHECKPOINT_FILE = "model/best.ckpt-118"
NEW_CHECKPOINT_FILE = "model2/best.ckpt-118"

import tensorflow as tf
vars_to_rename = {
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/B': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/B/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_0': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_0/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_F_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/w_f_diag',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_F_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_I_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/w_i_diag',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_I_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_O_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/w_o_diag',
    'BiRNN/BW/MultiRNNCell/Cell0/LSTMCell/W_O_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/B': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/bias',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/B/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/bias/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_0': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/kernel',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_0/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/kernel/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_F_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/w_f_diag',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_F_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/w_f_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_I_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/w_i_diag',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_I_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/w_i_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_O_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/w_o_diag',
    'BiRNN/BW/MultiRNNCell/Cell1/LSTMCell/W_O_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_1/lstm_cell/w_o_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/B': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/bias',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/B/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/bias/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_0': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/kernel',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_0/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/kernel/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_F_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/w_f_diag',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_F_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/w_f_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_I_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/w_i_diag',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_I_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/w_i_diag/Momentum',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_O_diag': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/w_o_diag',
    'BiRNN/BW/MultiRNNCell/Cell2/LSTMCell/W_O_diag/Momentum': 'bidirectional_rnn/bw/multi_rnn_cell/cell_2/lstm_cell/w_o_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/B': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/B/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_0': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_0/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_F_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/w_f_diag',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_F_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_I_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/w_i_diag',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_I_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_O_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/w_o_diag',
    'BiRNN/FW/MultiRNNCell/Cell0/LSTMCell/W_O_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/B': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/bias',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/B/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/bias/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_0': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/kernel',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_0/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/kernel/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_F_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/w_f_diag',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_F_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/w_f_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_I_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/w_i_diag',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_I_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/w_i_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_O_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/w_o_diag',
    'BiRNN/FW/MultiRNNCell/Cell1/LSTMCell/W_O_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_1/lstm_cell/w_o_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/B': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/bias',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/B/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/bias/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_0': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/kernel',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_0/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/kernel/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_F_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/w_f_diag',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_F_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/w_f_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_I_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/w_i_diag',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_I_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/w_i_diag/Momentum',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_O_diag': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/w_o_diag',
    'BiRNN/FW/MultiRNNCell/Cell2/LSTMCell/W_O_diag/Momentum': 'bidirectional_rnn/fw/multi_rnn_cell/cell_2/lstm_cell/w_o_diag/Momentum',
}
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
  if old_name in vars_to_rename:
    new_name = vars_to_rename[old_name]
  else:
    new_name = old_name
  new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
  sess.run(init)
  saver.save(sess, NEW_CHECKPOINT_FILE)
