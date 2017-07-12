# labels input as allosteric (A) or orthostreic (O)
# Uses NB model (nb.py)
# Usage: python predict_nb.py input_file.txt
# example: python predict_nb.py test_input/A_ASD0023_2_1N5M_1_desc.txt 
# writes output in the same directory as input file

# import stuff
import nb
import sys
#

# init an nb object
nb = nb.Nb() 

# input file
infile = sys.argv[1]

# parse input file
# parse_tsv assumes the input file has multiple lines (samples)
inheaders, indata = nb.parse_tsv(infile)

# predict the input
predictions =  nb.predict(nb.descs,nb.data,inheaders,indata)

#write output
outfile = infile.split('.')[0] + '_out.txt'
outcontent = ''
outs = [['prediction', 'name', 'pratio']] + predictions
for p in outs:
  content = ','.join([str(item) for item in p]) + '\n'
  outcontent += content
outfile = open(outfile, 'w')
outfile.write(outcontent)
outfile.close()


