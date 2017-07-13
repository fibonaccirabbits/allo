# a simple neural network implementation

#import stuff
import numpy as np
import sys


class Network():
  '''
  Define a network object
  '''
  def __init__(self,sizes):
    '''
    Initializes a network object.
    '''
    rstate = np.random.RandomState(0)
    self.sizes = sizes
    self.num_layers = len(sizes)
    self.biases = [rstate.randn(i,1) for i in sizes[1:]]
    self.weights = [rstate.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

  def SGD(self,training_data,epochs,eta,mini_batch_size,test_data=None):
    '''
    Performs stochastic gradient descent (SGD)

    input:
    ------
    training_data: list
      A list of training examples
    epochs: int
      A total number of iterations
    eta: float
      A learning rate
    mini_batch_size: int
      A size of a mini batch
    test_data: list
      A list of test examples
    '''
    n_train = len(training_data)
    if test_data: n_test = len(test_data)
    for epoch in range(epochs):
      np.random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in
                      range(0,n_train,mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch,eta)
      if test_data:
        result = self.evaluate(test_data)
        print 'Epoch {} completed: {}/{}'.format(epoch,result,n_test)
      else:
        print 'Epoch {} complted.'.format(epoch)

  def evaluate(self, test_data):
    '''
    Evaluates samples in test data

    input:
    ------
    test_data:
      A list test examples
    '''
    result = [int(np.argmax(self.feedforward(x)) == np.argmax(y)) for x,y in test_data] 
    result = sum(result)
    return result    



  def SGDR(self,training_data,epochs,eta,mini_batch_size,test_data=None):
    '''
    Performs stochastic gradient descent (SGD)

    input:
    ------
    training_data: list
      A list of training examples
    epochs: int
      A total number of iterations
    eta: float
      A learning rate
    mini_batch_size: int
      A size of a mini batch
    test_data: list
      A list of test examples
    '''
    n_train = len(training_data)
    if test_data: n_test = len(test_data)
    for epoch in range(epochs):
      np.random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in
                      range(0,n_train,mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch,eta)
      if test_data:
        result = self.evaluate_r(test_data)
        print 'Epoch {} completed: SSE {}'.format(epoch,result)
      else:
        print 'Epoch {} complted.'.format(epoch)

  def evaluate_r(self, test_data):
    '''
    Evaluates samples in test data. uses mean sqared error.

    input:
    ------
    test_data:
      A list test examples
    '''
    #result = [int(np.argmax(self.feedforward(x)) == np.argmax(y)) for x,y in test_data] 
    result = [(self.feedforward_r(x) - y)**2 for x,y in test_data] 
    result = np.sum(result)/len(test_data)
    return result    


  def SGD2(self, training_data,epochs,eta,mini_batch_size,test_data=None):
    '''
    Performs stochastic gradient descent.

    input:
    training_data: list
      A list of training examples
    epochs: int
      Number of iterations
    eta: float
      A learning rate
    mini_batch_size: int
      The size of a mini batch
    test_data: list
      A list of test examples
    
    '''

    if test_data: n_test = len(test_data)
    n_train = len(training_data)
    for epoch in range(epochs):
      np.random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in
                      range(0,n_train,mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        r0,r1  = self.evaluate2(test_data)
        print 'Epoch {} completed: {}/{} ; {}/{}'.format(epoch, r0[0],
                                                      r0[1], r1[0],r1[1])
      else:
        print 'Optimizing the net, epoch {}/{}'.format(epoch+1,epochs)
    print 'Done!'


  def SGD2_scan(self, training_data,epochs,eta,mini_batch_size,test_data):
    '''
    Performs stochastic gradient descent. Evaluates at every epoch.

    input:
    training_data: list
      A list of training examples
    epochs: int
      Number of iterations
    eta: float
      A learning rate
    mini_batch_size: int
      The size of a mini batch
    test_data: list
      A list of test examples
    
    '''

    if test_data: n_test = len(test_data)
    n_train = len(training_data)
    rs = []
    for epoch in range(epochs):
      np.random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in
                      range(0,n_train,mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      r0,r1  = self.evaluate2(test_data)
      r = [epoch] + r0 + r1
      rs.append(r)
      print 'Epoch {} completed.'.format(epoch)
    return rs

  def evaluate2(self,test_data):
    '''
    Evaluates the network. Uses test data.
    '''
    r0,r1 = [0,0], [0,0]
    for item in test_data:
      x,y = item
      yp = self.feedforward(x)
      y,yp = np.argmax(y), np.argmax(yp)
      if y == 0 and y==yp:
        r0[0] += 1
      if y == 0:
        r0[1] += 1
      if y == 1 and y == yp:
        r1[0] += 1
      if y == 1:
        r1[1] += 1
    return r0,r1


  def feedforward(self,x):
    '''
    Feedforward a sample through the network  
    
    input:
    x: array
      An array of values
    '''
    for b,w in zip(self.biases,self.weights):
      x = self.sigmoid(np.dot(w,x) + b)
    return x


  def feedforward_r(self,x):
    '''
    Feedforward a sample through the network. Does note activate the output
    node.  
    
    input:
    x: array
      An array of values
    '''
    lw = range(len(self.weights))
    lwend = lw[-1]
    for i,b,w in zip(lw,self.biases,self.weights):
      if i != lwend:
        x = self.sigmoid(np.dot(w,x) + b)
      else:
        x = np.dot(w,x) + b
    return x


  def update_mini_batch(self,mini_batch,eta):
    '''
    Updates biases and weights. Uses a mini batch

    input:
    ------
    mini_batch: list
      A list of mini batch examples
    eta: float
      A learning rate
      
    '''
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x,y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x,y)
      nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
      nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
    m = len(mini_batch)
    self.biases = [b - (eta*nb/m) for b,nb in zip(self.biases,nabla_b)]
    self.weights = [w - (eta*nw/m) for w,nw in zip(self.weights,nabla_w)]


  def backprop(self,x,y):
    '''
    Backpropagates a sample through the network
  
    input:
    ------
    x: array
      An array of input value
    y: array
      An array of output value
    '''
    delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
    delta_nabla_w = [np.zeros(w.shape) for w in self.biases]
    activation = x
    activations = [x]
    zs = []
    for b,w in zip(self.biases,self.weights):
      z = np.dot(w,activation) + b
      zs.append(z)
      activation = self.sigmoid(z)
      activations.append(activation)
    delta = self.cost_prime(activations[-1],y) * self.sigmoid_prime(zs[-1])
    delta_nabla_b[-1]= delta
    delta2 = np.dot(delta,activations[-2].transpose())
    delta_nabla_w[-1] = delta2
    for l in range(2,self.num_layers):
      z = zs[-l]
      sp = self.sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
      delta_nabla_b[-l] = delta
      delta2 = np.dot(delta,activations[-l-1].transpose())
      delta_nabla_w[-l] = delta2
    return delta_nabla_b, delta_nabla_w
  

  def cost_prime(self,output_activation,y):
    '''
    The first derivative of the cost function

    input:
    ------
    output_activation: array
      The final activation
    y: array
      An outpt array
    '''
    cp = output_activation - y
    return cp

  def sigmoid_prime(self,z):
    '''
    The first derivative of the sigmoid fucntion
    
    input:
    ------
    z: array
      An array of values
    '''
    sp  = self.sigmoid(z) * (1-self.sigmoid(z))
    return sp



  def sigmoid(self,z):
    '''
    A sigmoid function.
    '''
    s = 1.0/(1.0 + np.exp(-z))
    return s


# test stuff

def test_net():
  '''
  Tests stuff
  '''
  import data_loader
  d = data_loader.Loader()
  d.load()
  sizes = [d.x_dim,30,d.y_dim]
  net = Network(sizes)
  k = d.size/3
  training_data, test_data = d.data[k:], d.data[:k]
  net.SGD2(training_data,500,3.0,2,test_data=test_data)

def test_mnist():
  '''
  Tests a network on mnist data
  '''
  import mnist_loader
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  in_size = len(training_data[0][0])
  out_size = len(training_data[0][1])
  test_data2 = [] 
  for x,y in test_data:
    new_y  = np.zeros((out_size,1))
    new_y[y] = 1
    test_data2.append((x,new_y))
  sizes = [in_size,30,out_size]
  net = Network(sizes)
  net.SGD(training_data,30,3.0,10000,test_data=test_data2)

def test_net2():
  import data_loader
  d = data_loader.Loader()
  d.load2()
  np.random.shuffle(d.data)
  k = d.size/3
  data_r = [(x,y[4:]*10e10) for x,y in d.data ]
  training_data, test_data = data_r[k:], data_r[:k]
  sizes = [d.x_dim,len(training_data[0][1])]
  net = Network(sizes)
  net.SGDR(training_data,100,3.0,10,test_data=test_data) 
  results = [net.feedforward(x) for x,y in test_data]
  training_data_c = [(np.concatenate((x,y[4:])),y[:2]) for x,y in d.data[k:]]
  test_data_c = [(np.concatenate((x[0],y)),x[1][:2]) for x,y in zip(d.data[:k],results)]
  x_dim =  training_data_c[0][0].shape[0]
  y_dim = training_data_c[0][1].shape[0]
  #sizes_c = [x_dim,10,10,10,10,y_dim]
  sizes_c = [x_dim,y_dim]
  net2 = Network(sizes_c)
  net2.SGD2(training_data_c,200,0.1,5,test_data=test_data_c) 
  print 'sizes %s' % sizes
  print 'sizes_c %s ' % sizes_c



#test_mnist()
#test_net()
#test_net2()
sizes = [2,3]
#net = Network(sizes)
#net2 = Network(sizes)
#print net.biases
#print net2.biases
