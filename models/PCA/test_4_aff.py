import affinity_layer_pdl as a_pdl
import paddle as pdl
import numpy as np

layer = a_pdl.Affinity(2)
optim = pdl.optimizer.Adam(parameters=layer.parameters())

print(layer.parameters('A'))

X = pdl.to_tensor(np.arange(8,dtype=np.double).reshape(2,2,2),dtype='float32')
Y = pdl.to_tensor(np.arange(4,dtype=np.double).reshape(1,2,2),dtype='float32')

Z = layer.forward(X, Y)
Z.backward()

optim.step()
print("after back\n", layer.parameters())
