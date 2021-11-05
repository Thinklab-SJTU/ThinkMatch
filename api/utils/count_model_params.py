import numpy as np

def count_parameters(model):
  return np.sum(np.prod(v.size) for v in model.trainable_params() if "auxiliary" not in v.name)
