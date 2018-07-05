from data_utils import load_CIFAR10
import tensorflow as tf 
import numpy as np

# macros
CIFAR_10_DEPTH = 10
EPS = 1e-8
VALIDATION_CHECK = 100


#hyper parameters
TOT_ITER = 10000
BATCH_SIZE = 10
REG = 1.0
LEARNING_RATE = 0.01

def define_model(x):

  h1 = tf.contrib.layers.fully_connected(x,500,activation_fn=tf.nn.relu)
  h2 = tf.contrib.layers.fully_connected(h1,200,activation_fn=tf.nn.relu)
  scores = tf.layers.dense(h2,CIFAR_10_DEPTH,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(REG),use_bias=True)
  
  return scores

def define_loss(scores,y):


  Lclass = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=scores, 
    labels=y))

  
  Lreg = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  
  loss = Lclass + Lreg
  return loss
  
def get_accuracy(scores,y):
  
  correct_preds = tf.equal(tf.argmax(scores, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))
  return accuracy


  
def define_optimizer(loss,learning_rate = LEARNING_RATE):    

  optimizer=tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.minimize(loss)
  return train_op         


def main():

  # get the cifar-10 data
  cifar10_dir = './data'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir,flatten = True)


  
  # get the attributes of the input data
  train_images = X_train.shape[0]
  test_images = X_test.shape[0]
  img_size = X_train.shape[1]


  ### create a model ###
  
  # first we create a placeholder to input the data.
  x = tf.placeholder(tf.float32,shape=[None,img_size],name ='X')
  
  
  # we create a placeholder for our labeled data and make into a 
  y = tf.placeholder(tf.int32,shape=[None],name ='Y')   
  # we need the labels in one hot encoding
  y_hot = tf.one_hot(y,CIFAR_10_DEPTH)

  
  # define the model structure.
  scores = define_model(x)
  loss = define_loss(scores,y_hot)
  train_op = define_optimizer(loss)
  accuracy = get_accuracy(scores,y_hot)
  

  init = tf.global_variables_initializer()
  tot_loss = 0

  ### train model ###
  with tf.Session() as sess:
    
    # initialize variables
    sess.run(init)
    
    
    
    for i in range(TOT_ITER):
      
        
      # Grab the batch data... (handle last batch...)
      j = BATCH_SIZE*i % train_images
      if (j+BATCH_SIZE > train_images):
        X_batch = X_train[j:]
        y_batch = y_train[j:]            
      else:
        X_batch = X_train[j:j+BATCH_SIZE]
        y_batch = y_train[j:j+BATCH_SIZE]            
    

      feed_dict={x:X_batch,y:y_batch}      


      # do training on batch, return the summary and any results...
      _,train_loss = sess.run([train_op,loss],feed_dict)
      #print(train_loss)
      tot_loss += train_loss

          
      if (i % VALIDATION_CHECK == 0): 
        # get the test accuracy
        [train_acc,train_loss]=sess.run([accuracy,loss],feed_dict = feed_dict) 
        
        # get the test set accuracy
        
        # test a random sample of the test set.
        mask = np.arange(test_images)
        np.random.shuffle(mask)

        
        feed_dict={x:X_test[mask[0:BATCH_SIZE]],y:y_test[mask[0:BATCH_SIZE]]}   
        [test_acc]=sess.run([accuracy],feed_dict = feed_dict) 
    
        # read out current results.
        print('*** iteration %d ***' % (i))
        print('Loss %1.4f' % (tot_loss/VALIDATION_CHECK))
        print('Training accuracy = %1.2f' % (train_acc))
        print('Test accuracy = %1.2f ' % (test_acc))
        tot_loss = 0
            
            

    # grab final test accuracy
    feed_dict={x:X_test,y:y_test}   
    [test_acc]=sess.run([accuracy],feed_dict = feed_dict) 
    print('Final accuracy = %1.2f' % (test_acc))




  print('done')

  
  
  

if __name__ == '__main__':
    main()
    