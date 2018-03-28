# cnn6-mnist-noisy-signaling
# A 6-layer Convolutional neural nets class for MNIST recognition
# Support training, prediction, and extract features maps from the tensor graph
# Support saving/loading the model (i.e., all weights) to/from numpy arrays (.npy files)
# Add Gaussian noise after each matrix multiplication (i.e., after each signal coming out of convolutional and dense layers)
# Example usage of training from scratch
model = cnn6()  
model.build()
model.fit(X_train=mnist.train.images,
          Y_train=mnist.train.labels,
          X_test=mnist.test.images,
          Y_test=mnist.test.labels,
          num_epoch=1000000,    
          model_save_mode=True,
          model_save_path=model_path,
          batch_size = 32)
# Example usage of predicting from trained model ("cnn6.npy")
model = cnn6("cnn6.npy")  
model.build()
pred_labels = model.predict(X_test=mnist.test.images)
true_labels = np.argmax(mnist.test.labels,axis=1)
accuracy = np.sum((true_labels==pred_labels)+0.0)/len(true_labels)

# Example of plotting signals out of each neuron in each layer and drawing the feature maps
model = cnn6()
model.build()
model.load_variables(file_path='cnn6.npy')
model.extract_signaling(data=mnist.test.images, labels=mnist.test.labels)
plt.show()
