# Prioritizes (ranks) allosteric pocket in a set of pockets
# Uses nn.py, data_loader.py

# import stuff
import nn
import data_loader
import parse_tsv
import numpy as np
import sys
#

# Initialze a training data object.
d = data_loader.Loader()
d.load2()
trdata = [(x,y[:2]) for x,y,z in d.data]
d.y_headers  = d.y_headers[:2]

# Initialize a network object
sizes = [len(d.x_headers),10,10,10,10,10,len(d.y_headers)]
net = nn.Network(sizes)
training_data = trdata
epochs = 30
eta = 3.
mini_batch_size = 10

# optimize the biases and weights of the net
net.SGD2(training_data,epochs,eta,mini_batch_size)

# parse input file
infile = sys.argv[1] 
headers, data =  parse_tsv.parse_tsv(infile)

# rank samples
indices = [i for i,item in enumerate(headers) if item in d.x_headers] 
results = []
for datum in data:
  x = [item for i,item in enumerate(datum) if i in indices]
  x = np.array(x).reshape(len(x),1)
  x = (x-d.xmins)/(d.xmaxs-d.xmins)
  a = net.feedforward(x) 
  result = (datum[0],round(a[1],5))
  results.append(result)
results = sorted(results, key=lambda item: item[-1],reverse=True)

# write output file
outfile = infile.split('.')[0] + '_nn_out.txt'
#headers = ['name', 'predicted probability']
#contents = headers + results
outcontent = 'name,predicted probability\n'
for content in results:
  outcontent += ','.join([str(c) for c in content]) + '\n'
print 'Output will be written to: %s' % outfile
outfile = open(outfile ,'w')
outfile.write(outcontent)
outfile.close()




