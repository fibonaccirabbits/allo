#naive bayes stuff

#import stuff
import numpy as np
import sys
import scipy.stats as scstats
import math


class Nb():
  '''
  A naive bayes class.
  '''
  def __init__(self):
    '''
    Inits an NB object.
    '''
    self.datafile = '../data/ao.tsv' 
    self.headers, contents = self.parse_tsv(self.datafile)
    self.descs, self.data = self.nzdata(self.headers,contents)



  def split_a_o(self, headers, data):
    '''splits allosteric and orthosteric samples '''
    adata = []
    odata = []
    for datum in data:
      label = datum[0][0]
      if label == 'A':
        adata.append(datum)
      else:
        odata.append(datum)
    return headers, adata, odata


  def nzdata(self,headers,data):
    '''
    returns non zero data.
    '''
    data = np.array(data, dtype = 'object')
    s = data[:,0]
    l = data[:,3]
    x = np.ones((data.shape[0], data.shape[1]-2))
    x[:,:2]= data[:,1:3]
    x[:,2:] = data[:,4:]
    headers.pop(0)
    headers.pop(2)
    means = np.mean(x,axis=0)
    nonzeros = []
    for i,mean in enumerate(means):
      #if mean !=0:
      if mean >= 0.1:
        nonzeros.append(i)
    nz = np.ones((x.shape[0],len(nonzeros)+1),dtype='object')
    nzh = ['name']
    for i,i2 in enumerate(nonzeros):
      nz[:,i+1]=x[:,i2]
      nzh.append(headers[i2])
    nz[:,0] = s
    nz = nz.tolist()
    return nzh,nz


  def parse_tsv(self, tsvfile):
    '''parses a tab separated file in a given path. returns headers and
    contents. does not incluce lig_name in the newcontents'''
    contents = open(tsvfile).read().splitlines()
    headers = contents[0].split('\t')
    headers.pop(3)
    data = []
    for content in contents[1:]:
      newcontent = content.split('\t')
      newcontent.pop(3)
      floats = [float(item) for item in newcontent[1:]]
      samplename = newcontent[0]
      datum = [samplename] + floats
      data.append(datum)
    return headers, data



  def predict(self, trheaders, traindata,teheaders,testdata):
    '''builds a and o kde dicts from train data. predict a class fo each datum in test
    data '''
    agkdes, ogkdes, descs = self.data2gkdesdict(trheaders, traindata)
    aprior, oprior = self.get_prior(trheaders, traindata)
    predictions = []
    for datum in testdata:
      values = []
      adens, odens = [aprior], [oprior] # allo density, orth density
      samplename = datum[0]
      for i, value in enumerate(datum):
        desc = teheaders[i]
        if desc in descs and desc != 'name':
          values.append(value)
          agkde = agkdes[desc] 
          ogkde = ogkdes[desc]
          aden = agkde.evaluate(value)[0]
          oden = ogkde.evaluate(value)[0]
          if aden >= 10e-16 and oden >= 10e-16:
            adens.append(aden)
            odens.append(oden)
      aproduct = self.log_product(adens)
      oproduct = self.log_product(odens)
      pratio = aproduct/oproduct
      if pratio < 1.:
        pred_class = 'A'
      else:
        pred_class = 'O'
      predictions.append([pred_class, samplename, round(pratio,3)])
    return predictions

  def log_product(self, values):
    '''returns the log of a product '''
    sumlog = 0
    for value in values:
      logvalue = math.log(value) 
      sumlog += logvalue
    return sumlog

  def get_prior(self, headers, data):
    '''returns priors (aprior, oprior) fro the given data '''
    headers, adata, odata = self.split_a_o(headers, data) 
    nadata, nodata = len(adata), len(odata) # number of samples (n)
    ntotal = nadata + nodata
    aprior, oprior = nadata/float(ntotal), nodata/float(ntotal)
    return aprior, oprior


  def data2gkdesdict(self, headers, data):
    '''returns a cumulative desity dictionaries (agkdes and ogkdes) and list of
    descriptor (descs) for a given data'''
    headers , adata, odata = self.split_a_o(headers, data)
    aheaders, acolumndata = self.rows2columns(headers, adata)
    oheaders, ocolumndata = self.rows2columns(headers, odata)
    acolumn_dict = self.column_dict(aheaders, acolumndata)
    ocolumn_dict = self.column_dict(oheaders, ocolumndata)
    agkdes = self.gkde_dict(aheaders, acolumndata)
    ogkdes = self.gkde_dict(oheaders, ocolumndata)
    lena = len(agkdes)
    leno = len(ogkdes)
    if lena < leno:
      descs = agkdes.keys()
    else:
      descs = ogkdes.keys()
    return agkdes, ogkdes, descs 


  
  def rows2columns(self, headers, rowdata):
    '''transforms rowdata to column data'''
    desclen = len(headers) # descriptors length
    columndata = [[] for i in range(desclen)]
    for rowdatum in rowdata:
      for i in range(desclen):
        rowitem = rowdatum[i]
        columndata[i].append(rowitem)
    return headers, columndata


  def gkde_dict(self, headers, columndata): 
    '''returns gkde dictionary from a given columndata. Discards singular
    matrix columndatum'''
    gkdes = {}
    exceptions = []
    for i, header in enumerate(headers):
      columndatum = columndata[i]
      try:
        gkde = self.get_gkde(columndatum)
        gkdes[header] = gkde
      except :
        e = sys.exc_info()
        gkdes[header] = e
    gkdes2 = {} # holds descriptors with gkde exclusively
    for gkde in gkdes:
      values = gkdes[gkde]
      valtype = type(values)
      if type(values) is scstats.kde.gaussian_kde:
        gkdes2[gkde] = values
      elif gkde == 'name':
        gkdes2[gkde] = values
    return gkdes2


  def get_gkde(self, columndatum):
    ''' returns a gaussian kernel density estimator for a given columnddatum'''
    gkde = scstats.gaussian_kde(columndatum) 
    return gkde


  def column_dict(self, headers, columndata):
    '''returns column dictionary. Keys are descriptors name, values are the
    columns '''
    column_dict = {}
    for i, header in enumerate(headers):
      values = columndata[i]
      key = header
      column_dict[key] = values
    return column_dict




# run stuff
#nb = Nb()
#nb.predict(nb.descs,nb.data[20:],nb.data[:20])



