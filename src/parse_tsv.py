# parses a tab separated file 


def parse_tsv(tsvfile):
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
