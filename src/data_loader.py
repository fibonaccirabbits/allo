# A data loader.

#import stuff
import numpy as np
import sys
import os


class Loader():
  '''
  A data loader object
  '''
  def __init__(self):
    self.data = []
    self.path = ''
    self.size = 0
    self.x_dim = 0
    self.y_dim = 0
    self.y_headers = ''
    self.x_headers = ''
    self.xmins = []
    self.xmaxs = []

  def load(self):
    '''
    Parses a file

    input:
    ------
    filepath: string
      A file path

    output:
    -------
    '''
    #filepath = '../data/Core_Diversity_Set_Chains_All.tsv'
    filepath = '/'.join(os.getcwd().split('/')[:-1]) + '/data/Core_Diversity_Set_chains_all_clean.tsv'
    contents = open(filepath).read().splitlines()
    headers  = contents[0].split()
    y = []
    for content in contents[1:]:
      parts = content.split()
      ligname = parts[3]
      if ligname == 'non':
        y.append(1)
      else: 
        y.append(0)
    
    y = np.array(y).reshape(len(y),1)
    y2 =  1 - y
    Y = np.concatenate((y,y2),axis=1)
    X = [c.split()[5:] for c in contents[1:]]
    X = np.array(X).astype(float)
    Xn = np.copy(X)
    xmean = np.mean(Xn,axis=0)
    z_indices  = [i for i in range(len(xmean)) if xmean[i] == 0]
    NZ = np.delete(Xn,z_indices,1)
    xmax = np.max(NZ,axis=0)
    xmin = np.min(NZ,axis=0)
    NZ = (NZ-xmin)/(xmax-xmin)
    data = [(x.reshape(len(x),1), y.reshape(len(y),1)) for x,y in zip(NZ,Y)]
    self.data = data
    self.path = filepath
    self.size = len(data)
    self.x_dim = len(data[0][0])
    self.y_dim = len(data[0][1])


  def load2(self):
    '''
    Loads data. Uses ligand features as Y.
    '''
    filepath = '../data/aplc.tsv'
    contents = open(filepath).read().splitlines()
    y = []
    y_headers = ['orto', 'allo']
    for content in contents[1:]:
      parts = content.split()
      ligname = parts[3]
      if ligname == 'non':
        y.append(1)
      else: 
        y.append(0) 
    y = np.array(y).reshape(len(y),1)
    y2 = 1-y
    Y = np.concatenate((y,y2),axis=1)
    str_headers = [0,3]
    n_feat = len(contents[0])
    n = len(contents)-1 # exclude headers
    contents = [content.split('\t') for content in contents] 
    X = np.array(contents)
    Z = np.copy(X[1:,0])
    X = np.delete(X,str_headers,1)
    y_headers = y_headers + X[0,:3].tolist()
    x_headers = X[0,3:].tolist()
    X = np.delete(X,0,0)
    YL = np.zeros((n,3))
    YL[:,:] = X[:,:3].astype('float')
    X = np.delete(X,[0,1,2],1).astype('float')
    means = np.mean(X,axis=0)
    izeros = [i for i in range(len(means)) if means[i]  == 0]
    X = np.delete(X,izeros,axis=1)
    mins = np.min(X,axis=0)
    maxs = np.max(X,axis=0)
    means = np.mean(X,axis=0)
    x_headers = np.delete(x_headers,izeros).tolist()
    X = (X-mins)/(maxs-mins)
    Y = np.concatenate((Y,YL),axis=1)
    data = [(x.reshape(X.shape[1],1),y.reshape(Y.shape[1],1),z) for x,y,z in zip(X,Y,Z)]
    self.data = data
    self.size = len(data)
    self.x_dim = X.shape[1]
    self.y_dim = Y.shape[1]
    self.y_headers = y_headers
    self.x_headers = x_headers
    self.path = filepath
    self.xmins = mins.reshape(len(mins,),1)
    self.xmaxs = maxs.reshape(len(maxs,),1)

# test stuff

#d = Loader()
#d.load()
