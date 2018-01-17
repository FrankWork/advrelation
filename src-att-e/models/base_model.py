import os
import sys
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

flags = tf.app.flags
flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

FLAGS = tf.app.flags.FLAGS


class BaseModel(object):

  def __init__(self):
    self.tensors = dict()

  def set_saver(self, save_dir):
    '''
    Args:
      save_dir: relative path to FLAGS.logdir
    '''
    # shared between train and valid model instance
    self.saver = tf.train.Saver(var_list=None)
    self.save_dir = os.path.join(FLAGS.logdir, save_dir)
    self.save_path = os.path.join(self.save_dir, "model.ckpt")

  def restore(self, session):
    ckpt = tf.train.get_checkpoint_state(self.save_dir)
    self.saver.restore(session, ckpt.model_checkpoint_path)

  def save(self, session, global_step):
    self.saver.save(session, self.save_path, global_step)

  def body(self, data):
    raise NotImplementedError

  def loss(self, logits, labels):
    raise NotImplementedError
  
  def prediction(self, logits):
    raise NotImplementedError
  
  def accuracy(self, logits, labels):
    raise NotImplementedError
  
  def set_train_mode(self):
    raise NotImplementedError
  
  def set_test_mode(self):
    raise NotImplementedError

  def evaluate(self, eval_steps, test_data):
    self.set_test_mode()
    moving_acc = 0
    n_step = 0

    for batch_data in tfe.Iterator(test_data):
      labels, logits = self.forward(batch_data)
      acc = self.accuracy(logits, labels)
      moving_acc += acc
      n_step += 1
      if n_step % eval_steps == 0:
        break
    
    self.set_train_mode()
    return moving_acc / n_step

  def train_and_eval(self, num_epochs, num_batchs_per_epoch, 
                      lrn_rate, train_data, test_data):
    best_acc, best_epoch = 0., 0
    start_time = time.time()
    orig_begin_time = start_time
    
    val_and_grad_fn = tfe.implicit_value_and_gradients(self.loss)
    # grad_fn = tfe.implicit_gradients(self.loss)

    optimizer = tf.train.AdamOptimizer(lrn_rate)

    epoch = 0
    moving_loss, moving_acc = 0, 0
    max_norm = 0
    device = "/gpu:0" if tfe.num_gpus() > 1 else "/cpu:0"
    
    with tf.device(device):
      for batch, batch_data in enumerate(tfe.Iterator(train_data)):
        loss, grad_and_var = val_and_grad_fn(batch_data)
        
        # grad_list = [grad for grad, _ in grad_and_var]
        # max_norm = max(max_norm, tf.global_norm(grad_list)) # max_norm < 2
        
        acc = self.tensors['acc']

        optimizer.apply_gradients(grad_and_var)
        # print(batch, loss.numpy(), acc.numpy())

        moving_loss += loss
        moving_acc += acc

        if (batch+1) % num_batchs_per_epoch == 0:
          moving_loss /= num_batchs_per_epoch
          moving_acc /= num_batchs_per_epoch

          # epoch duration
          now = time.time()
          duration = now - start_time
          start_time = now
          
          valid_acc = self.evaluate(28, test_data)
          if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch
          
          # var_list = [var for _, var in grad_and_var]
          # norm_list = [tf.norm(var) for var in var_list]
          # for var, norm in zip(var_list, norm_list):
          #   print('%s\t%.2f' % (var.name, norm.numpy()))

          print("Epoch %d loss %.2f acc %.2f %.4f time %.2f" % 
              (epoch, moving_loss, moving_acc, valid_acc, duration))
          sys.stdout.flush()

          epoch += 1
          moving_loss = 0
          moving_acc = 0
          # max_norm = 0
          if epoch == num_epochs:
            break
    
    duration = time.time() - orig_begin_time
    duration /= 3600
    print('Done training, best_epoch: %d, best_acc: %.4f' % (best_epoch, best_acc))
    print('duration: %.2f hours' % duration)
    sys.stdout.flush()
  
def conv_block_v2(inputs, kernel_size, num_filters, name, training, 
               batch_norm=False, initializer=None, shortcut=None, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    conv_out = tf.layers.conv1d(inputs, num_filters, kernel_size,
                  strides=1, padding='same',
                  kernel_initializer=initializer,
                  name='conv-%d' % kernel_size,
                  reuse=reuse)
    if batch_norm:
      conv_out = tf.layers.batch_normalization(conv_out, training=training)
    conv_out = tf.nn.relu(conv_out)
    if shortcut is not None:
      conv_out = conv_out + shortcut
    return conv_out

def optimize(loss, lrn_rate, max_norm=None, decay_steps=None):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch_norm
  with tf.control_dependencies(update_ops):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    if decay_steps is not None:
      lrn_rate = tf.train.exponential_decay(lrn_rate, global_step, 
                                    decay_steps, 0.95, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(lrn_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    
    if max_norm is not None:
      gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    return train_op

