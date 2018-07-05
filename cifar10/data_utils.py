#import _pickle as pickle
import  pickle
import numpy as np
import os
import download



DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def maybe_download_and_extract(ROOT):
    """
    Download and extract the CIFAR-10 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """

    download.maybe_download_and_extract(url=DATA_URL, download_dir=ROOT)



def load_CIFAR_batch(filename,flatten = True):
  """ load single batch of cifar """
  
  print(filename)
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding = 'bytes')
    #print('dictionary key =',datadict.keys())
    X = datadict[b'data']/255
    Y = datadict[b'labels']
  
    if flatten == False:
      X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
      
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT,flatten = True):
  """ load all of cifar """
  xs = []
  ys = []
  print('Loading CIFAR10 Data from dir:',ROOT)
  
  maybe_download_and_extract(ROOT)
  for b in range(1,6):
    f = os.path.join(ROOT,'cifar-10-batches-py' ,'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    

  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT,'cifar-10-batches-py' ,'test_batch'))
  
  print('Done Load.')
  return Xtr, Ytr, Xte, Yte


