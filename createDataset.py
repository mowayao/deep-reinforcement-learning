

import h5py




f = h5py.File('../memory.hdf5', 'w')

f.create_dataset("phi",(1000000,4,84,84))


f.create_dataset("action",(1000000,),dtype='int')


f.create_dataset("phis",(1000000,4,84,84))

f.create_dataset("reward",(1000000,),dtype='float')


#print f['phi']

