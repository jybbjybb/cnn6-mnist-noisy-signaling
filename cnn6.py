import numpy as np
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


class cnn6:
    def __init__(self, dict_var_npy_path=None, keep_prob=0.5, n_classes=10, std_train=0.0, std_test=0.0):
        self.keep_prob = keep_prob
        self.n_classes = n_classes

        self.x = tf.placeholder(tf.float32,[None,28*28])
        self.y = tf.placeholder(tf.float32,shape=[None,self.n_classes])
        self.train_mode = tf.placeholder(tf.bool)
        
        self.std = tf.placeholder(tf.float32)
        self.std_train = std_train
        self.std_test = std_test

        #with tf.variable_scope("cnn6", reuse=tf.AUTO_REUSE):
        self.var_dict = {
            'conv1_weights': tf.get_variable(name="conv1_weights",shape=[3,3,1,32],initializer=tf.contrib.layers.xavier_initializer()),
            'conv1_biases': tf.get_variable(name="conv1_biases", shape=[32],initializer=tf.zeros_initializer()),
            'conv2_weights': tf.get_variable(name="conv2_weights",shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_biases': tf.get_variable(name="conv2_biases", shape=[32],initializer=tf.zeros_initializer()),
            'conv3_weights': tf.get_variable(name="conv3_weights",shape=[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_biases': tf.get_variable(name="conv3_biases", shape=[64],initializer=tf.zeros_initializer()),
            'conv4_weights': tf.get_variable(name="conv4_weights",shape=[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_biases': tf.get_variable(name="conv4_biases", shape=[64],initializer=tf.zeros_initializer()),
            'fc1_weights': tf.get_variable(name="fc1_weights", shape=[4*4*64,256],initializer=tf.contrib.layers.xavier_initializer()),
            'fc1_biases': tf.get_variable(name="fc1_biases",shape=[256],initializer=tf.zeros_initializer()),
            'fc2_weights': tf.get_variable(name="fc2_weights", shape=[256,10],initializer=tf.contrib.layers.xavier_initializer()),
            'fc2_biases': tf.get_variable(name="fc2_biases",shape=[10],initializer=tf.zeros_initializer())
        }
        self.padding = 'VALID'

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if (dict_var_npy_path is not None) and (os.path.exists(dict_var_npy_path)):
            self.use_existing_model = True
            self.model_path = dict_var_npy_path
        else:
            self.use_existing_model = False
            self.model_path = None

    def build(self):
        try:
            self.input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        except ValueError:
            print("Input image not having 784 pixels!")
        # Conv, pooling layers
        self.conv1_clean = tf.nn.conv2d(input=self.input_layer, filter=self.var_dict['conv1_weights'], strides=[1,1,1,1],padding=self.padding, name='conv1')
        self.conv1_clean = self.conv1_clean + self.var_dict['conv1_biases']
        self.conv1_noisy = self.gaussian_noise_layer(self.conv1_clean, self.std)    
        self.conv1_out = tf.nn.relu(self.conv1_noisy)

        self.conv2_clean = tf.nn.conv2d(input=self.conv1_out, filter=self.var_dict['conv2_weights'], strides=[1,1,1,1], padding=self.padding, name='conv2')
        self.conv2_clean = self.conv2_clean + self.var_dict['conv2_biases']
        self.conv2_noisy = self.gaussian_noise_layer(self.conv2_clean, self.std)
        self.conv2_out = tf.nn.relu(self.conv2_noisy)

        self.pool1 = tf.nn.max_pool(self.conv2_out,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='maxpool1')

        self.conv3_clean = tf.nn.conv2d(input=self.pool1, filter=self.var_dict['conv3_weights'], strides=[1,1,1,1],padding=self.padding, name='conv3')
        self.conv3_clean = self.conv3_clean + self.var_dict['conv3_biases']
        self.conv3_noisy = self.gaussian_noise_layer(self.conv3_clean, self.std)
        self.conv3_out = tf.nn.relu(self.conv3_noisy)

        self.conv4_clean = tf.nn.conv2d(input=self.conv3_out, filter=self.var_dict['conv4_weights'], strides=[1,1,1,1],padding=self.padding, name='conv4')
        self.conv4_clean = self.conv4_clean + self.var_dict['conv4_biases']
        self.conv4_noisy = self.gaussian_noise_layer(self.conv4_clean, self.std)
        self.conv4_out = tf.nn.relu(self.conv4_noisy)

        self.pool2 = tf.nn.max_pool(self.conv4_out,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='maxpool2')

        # Flatten conv layer
        self.flat = tf.reshape(self.pool2, [-1, 4*4*64])
        # Fully connected layers
        self.fc1_clean = tf.matmul(self.flat, self.var_dict['fc1_weights'])
        self.fc1_clean = self.fc1_clean + self.var_dict['fc1_biases']
        self.fc1_noisy = self.gaussian_noise_layer(self.fc1_clean, self.std)
        self.fc1_out = tf.nn.relu(self.fc1_noisy)
        self.dropout_layer = tf.cond(self.train_mode, lambda: tf.layers.dropout(inputs=self.fc1_out,rate=1.0-self.keep_prob), lambda: self.fc1_out)

        self.logits_clean = tf.matmul(self.dropout_layer, self.var_dict['fc2_weights']) + self.var_dict['fc2_biases']
        self.logits_noisy = self.gaussian_noise_layer(self.logits_clean, self.std)
        self.pred_prob = tf.nn.softmax(self.logits_noisy)


    def predict(self, X_test, model_path=None, sess_path=None):
        if model_path is not None:
            self.load_variables(file_path=model_path)
        elif sess_path is not None:
            saver = tf.train.Saver(max_to_keep=None)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, sess_path)

        if self.use_existing_model:
            print("Use existing model: "+self.model_path+". Loading...")
            self.load_variables(self.model_path)

        preds = self.sess.run(self.pred_prob, feed_dict={self.x:X_test, self.train_mode:False, self.std:self.std_test})

        return np.argmax(preds, axis=1)

    def fit(self,X_train,Y_train,X_test,Y_test,
            learning_rate=1e-4,
            num_epoch=1000000,
            sess_save_mode=False,
            sess_save_path='/tmp/mnist_tao',
            model_save_mode=True,
            model_save_path='./mnist_model_cnn6.npy',
            batch_size = 32):

        self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_noisy, labels=self.y)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,axis=1), tf.argmax(self.logits_noisy,axis=1)),"float"))
        if sess_save_mode:
            saver = tf.train.Saver(max_to_keep=None)

        if self.use_existing_model:
            print("Use existing model: "+self.model_path+". Loading...")
            self.load_variables(self.model_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        max_accu = 0.0
        num_iter_not_improved = 0
        num_iter_per_test = 100

        for i in range(num_epoch+1):
            if num_iter_not_improved > 20000:
                    break 

            x_train_batch, y_train_batch = self.random_next_batch(X_train, Y_train, batch_size)

            self.sess.run(self.train_step, feed_dict={self.x:x_train_batch, self.y:y_train_batch, self.train_mode:True, self.std:self.std_train})
        
            if i % num_iter_per_test == 0:
                accu_train = self.sess.run(self.accuracy, feed_dict={self.x:x_train_batch, self.y:y_train_batch, self.train_mode:False, self.std:self.std_train})
                accu_test = self.sess.run(self.accuracy, feed_dict={self.x:X_test, self.y:Y_test, self.train_mode:False, self.std:self.std_train})
                print("Iter:{0}, train_std={1}, train accu:{2}, test accu:{3}".format(i,self.std_train, accu_train,accu_test))
                if accu_test>max_accu:
                    num_iter_not_improved = 0
                    if sess_save_mode:
                        saver.save(self.sess,sess_save_path)
                        print('     Improved. Session saved at {0}'.format(sess_save_path))
                    else:
                        print('     Improved. Session save mode is False.')
                    if model_save_mode:
                        self.save_variables(file_path=model_save_path)
                        print('     Improved. Model saved at {0}'.format(model_save_path))
                    else:
                        print('     Improved. Model save mode is False')
                    max_accu = accu_test
                else:
                    num_iter_not_improved += num_iter_per_test
                    print('     Not Improved. {0} iters since last improve. Best accu: {1}'.format(num_iter_not_improved, max_accu))
                    

    def extract_signaling(self, data, labels, show_hist=True, show_feature_maps=True):
        print("Test on std:{0}".format(self.std_test))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,axis=1), tf.argmax(self.logits_noisy,axis=1)),"float"))
        
        accuracy, conv1_clean, conv2_clean, conv3_clean, conv4_clean, fc1_clean, logits_clean, conv1_noisy, conv2_noisy, conv3_noisy, conv4_noisy, fc1_noisy, logits_noisy = self.sess.run([self.accuracy, self.conv1_clean, self.conv2_clean, self.conv3_clean, self.conv4_clean, self.fc1_clean, self.logits_clean, self.conv1_noisy, self.conv2_noisy, self.conv3_noisy, self.conv4_noisy, self.fc1_noisy, self.logits_noisy],feed_dict={self.x:data, self.y:labels, self.train_mode:False, self.std:self.std_test} )
        print("Signals extracted, waiting to plot")

        layers_clean = [conv1_clean, conv2_clean, conv3_clean, conv4_clean, fc1_clean, logits_clean]
        layers_noisy = [conv1_noisy, conv2_noisy, conv3_noisy, conv4_noisy, fc1_noisy, logits_noisy]
        
        # calculate signal power
        num_signals = []
        power = []
        e_power = []
        for i in range(len(layers_clean)):
            layer = layers_clean[i]
            this_power = np.sum(np.square(layer))
            power.append(this_power)
            num_signals.append(layer.size)
            e_power.append(this_power/layer.size)
            print("Expected signal power:{0} of layer {1}".format(e_power[i],i))

        signal_power = sum(power)/sum(num_signals)
        print("Expected signal power overall: {0}".format(signal_power))

        self.plot_signal_histogram(layers_clean, layers_noisy, accuracy)
        
        if show_feature_maps and len(data)==1:
            self.draw_feature_maps(layers_clean=layers_clean, layers_noisy=layers_noisy)
        
        plt.show()

    def draw_feature_maps(self, layers_clean, layers_noisy, show_now=False):

        if self.std_test > 0:
            noise = []
            num_plots = 4
            bin_size = 0.1
            plt.figure()
            for i in range(num_plots):  
                plt.subplot(num_plots+1,1, i+1)
                self.plot_hist(layers_clean[i]-layers_noisy[i], bin_size=bin_size)        
            mu = 0;
            s = self.std_test     
            xx = np.linspace(mu - 5*s, mu + 5*s, 10*s/bin_size)
            yy = mlab.normpdf(xx, mu, s)
            yy = yy/np.sum(yy)
            plt.subplot(num_plots+1,1, num_plots+1)
            plt.plot(xx,yy,'b--')
            plt.grid()

        for layer in range(4):
            feature = layers_clean[layer]
            plt.figure() 
            print(feature.shape)
            for i in range(feature.shape[3]):
                n_rows = int(np.sqrt(feature.shape[3]))
                n_cols = int(np.ceil( (feature.shape[3]+0.0)/n_rows) )
                plt.subplot(n_rows, n_cols, i+1)
                plt.imshow(feature[0,:,:,i])
            plt.suptitle("Conv_"+str(layer+1)+"_clean")
                
        for layer in range(4):
            feature = layers_noisy[layer]
            plt.figure() 
            print(feature.shape)
            for i in range(feature.shape[3]):
                n_rows = int(np.sqrt(feature.shape[3]))
                n_cols = int(np.ceil( (feature.shape[3]+0.0)/n_rows) )
                plt.subplot(n_rows, n_cols, i+1)
                plt.imshow(feature[0,:,:,i])
            plt.suptitle("Conv_"+str(layer+1)+"_noisy")

        if show_now:
            plt.show()
        
        
    def plot_hist(self, XX, show_plot=False, bin_size=-1):
        X = np.asarray(XX).flatten()
        maxi = np.max(X)
        mini = np.min(X)
        if bin_size > 0:
            hist = np.histogram(X, bins=int((maxi-mini)/bin_size))
        else:
            hist = np.histogram(X)
        xx = (hist[1][:-1] + hist[1][1:])/2
        yy = hist[0]
        yy = (0.0+yy)/sum(yy)
        #print(xx,'\n',yy)
        plt.plot(xx,yy)
        plt.grid()
        if show_plot:
            plt.show()


    def plot_signal_histogram(self, layers_clean, layers_noisy, accuracy, show_now=False):
        # Two subplots, the axes array is 1-d
            fig, ax = plt.subplots(nrows=2,ncols=2)
            bin_size = 0.05
            _xlim = [-10,10]
            _ylim = [0,1]
            _yscale = 'log'

            plt.subplot(2,2,1)   
            for z in range(len(layers_clean)):
                layer_clean_flat = layers_clean[z].flatten()
                print("No. signals in layer {0}:{1}".format(z,layer_clean_flat.shape) )
                maxx = np.max(layer_clean_flat)
                minn = np.min(layer_clean_flat)
                num_bins = int((maxx-minn)/bin_size)
                if num_bins > 1:
                    hist = np.histogram(layer_clean_flat, bins=num_bins)
                else:
                    hist = np.histogram(layer_clean_flat)
                print("No. bins in layer{0}: {1}".format(z,num_bins))
                xx = (hist[1][:-1] + hist[1][1:])/2
                yy = (0.0+hist[0])/np.sum(hist[0])
                plt.plot(xx,yy) 
           
            # plot the noise level
            if self.std_test > 0:
                mu = 0;
                s = self.std_test
                xx = np.linspace(mu - 5*s, mu + 5*s, 5*s/bin_size)
                yy = mlab.normpdf(xx, mu, s)
                yy = yy/np.sum(yy)
                plt.plot(xx,yy,'b--')
            
            plt.legend(['conv1','conv2','conv3','conv4','fc1','out','sigma={0}'.format(self.std_test)])
            
            plt.xlim(_xlim)
            plt.yscale(_yscale)
            plt.title('Before Noise Injection, bin size={0}'.format(bin_size))
            plt.grid()
              
            plt.subplot(2,2,2)
            for z in range(len(layers_noisy)):
                layer_noisy_flat = layers_noisy[z].flatten()
                print("No. signals in layer {0}:{1}".format(z,layer_noisy_flat.shape) )
                maxx = np.max(layer_noisy_flat)
                minn = np.min(layer_noisy_flat)
                num_bins = int((maxx-minn)/bin_size)  
                if num_bins > 1:
                    hist = np.histogram(layer_noisy_flat,bins=num_bins)
                else:
                    hist = np.histogram(layer_noisy_flat)
                print("No. bins in layer{0}: {1}".format(z,num_bins))
                xx = (hist[1][:-1] + hist[1][1:])/2
                yy = (0.0+hist[0])/np.sum(hist[0])
                plt.plot(xx,yy) 
             
            plt.legend(['conv1','conv2','conv3','conv4','fc1','out'])
            plt.xlim(_xlim)
            
            plt.yscale(_yscale)
            plt.title('After Noise Injection, bin size={0}'.format(bin_size))
            plt.grid()
            
            plt.subplot(2,2,3)
            for k in range(10):
                this_output_neuron = layers_clean[-1][:,k]     
                hist = np.histogram(this_output_neuron)
                print("No. bins in Neuron #{0}: {1}".format(k,num_bins))
                xx = (hist[1][:-1] + hist[1][1:])/2
                yy = (0.0+hist[0])/np.sum(hist[0])
                plt.plot(xx,yy) 
            plt.legend(['0','1','2','3','4','5','6','7','8','9'])
            plt.yscale(_yscale)
            plt.title('10 Output Signals Before Noise Injection, bin size={0}'.format(bin_size))
            plt.grid()
    
            plt.subplot(2,2,4)
            for k in range(10):
                this_output_neuron = layers_noisy[-1][:,k]
                hist = np.histogram(this_output_neuron)
                print("No. bins in Neuron #{0}: {1}".format(k,num_bins))
                xx = (hist[1][:-1] + hist[1][1:])/2
                yy = (0.0+hist[0])/np.sum(hist[0])
                plt.plot(xx,yy) 
            plt.legend(['0','1','2','3','4','5','6','7','8','9'])
            plt.yscale(_yscale)
            plt.title('10 Output Signals After Noise Injection, bin size={0}'.format(bin_size))
            plt.grid()        
            plt.draw() 
            plt.pause(0.01)   
                      
            plt.suptitle('MNIST Trained std={0}, Tested std={1}, acc={2}'.format(self.std_train, self.std_test, accuracy))
            if show_now:
                plt.show()
            

    def get_variables(self):
        np_var_dict = self.sess.run(self.var_dict)
        return np_var_dict

    def summary(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print("{0}, {1} trainable parameters.".format(variable, variable_parameters))
            total_parameters += variable_parameters
        print("Total No. Parameters: {0}".format(total_parameters))


    def save_variables(self, file_path='./mnist_model_cnn6.npy'):
        var_dict = self.get_variables()
        np.save(file_path,var_dict)
        return True

    def load_variables(self,file_path='./mnist_model_cnn6.npy'):
        print("Loading parameters from "+file_path+"...")
        np_var_dict = np.load(file_path).item()
        self.model_path = file_path
        self.sess.run(tf.global_variables_initializer())
        for key in self.var_dict:
            self.sess.run(self.var_dict[key].assign(np_var_dict[key]))

    def close_session(self):
        self.sess.close()

    def random_next_batch(self, x, y, batch_size=1):
        idx = np.random.choice(x.shape[0],batch_size)
        x_batch = x[idx,...]
        y_batch = y[idx,...]
        return x_batch, y_batch

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise

    def set_train_std(self, std_train):
        self.std_train = std_train

    def set_test_std(self, std_test):
        self.std_test = std_test


















