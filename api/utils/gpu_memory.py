from pynvml import *
nvmlInit()
import torch

def gpu_free_memory(device_id):
    """
    Return total amount of available memory in Bytes
    :param device_id: GPU device id (int)
    :return: total amount of available memory in Bytes
    """
    #ree = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id)).free
    #rsvd = torch.cuda.memory_reserved(device_id)
    #used = torch.cuda.memory_allocated(device_id)
    torch.cuda.empty_cache()
    free = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id)).free
    return free