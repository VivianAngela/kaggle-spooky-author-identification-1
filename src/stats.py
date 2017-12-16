import csv
csv.field_size_limit(2147483647)

def stats():
  author_vs_text = {'EAP': [], 'HPL': [], 'MWS': []}
  with open('train.csv') as inputfile:
    t = []
    reader = csv.reader(inputfile)
    records = [record for record in reader]
    for record in records[1:]:
      author = record[2]
      text = record[1]
      id = record[0]
      if not author_vs_text.has_key(author):
        author_vs_text[author] = []
      author_vs_text[author].append((text,id))
    print 'Total Count: ', len(records)
  print 'Author\tCount\tCMaxLen\tCMaxID\tCMinLen\tCMinID\tWMaxLen\tWMaxID\tWMinLen\tWMinID'
  cavg=0
  wavg=0
  vocab = set()
  for author,texts in author_vs_text.iteritems():
    cmaxlen = 0
    cmaxid = 0

    cminlen = 99999999 
    cminid = 0

    wmaxlen = 0
    wmaxid = 0

    wminlen = 99999999
    wminid = 0

    for text in texts:
      cl = len(text[0])
      tokens = text[0].split(' ')
      for token in tokens:
        vocab.add(token)
      wl = len(tokens)
      cavg+=cl
      wavg+=wl
      if cl > cmaxlen:
        cmaxlen = cl
        cmaxid = text[1]
      elif cl < cminlen:
        cminlen = cl
        cminid = text[1]
      if wl > wmaxlen:
        wmaxlen = wl
        wmaxid = text[1]
      elif wl < wminlen:
        wminlen = wl
        wminid = text[1]
    print author + '\t' + str(len(texts)) + '\t' + str(cmaxlen) + '\t' + str(cmaxid) + '\t' + str(cminlen) + '\t' + str(cminid) + '\t' + str(wmaxlen) + '\t' + str(wmaxid) + '\t' + str(wminlen) + '\t' + str(wminid)
  print 'C Avg Len: ', cavg/19579
  print 'W Avg Len: ', wavg/19579
  print 'Total Unique Words: ', len(vocab)

stats()
