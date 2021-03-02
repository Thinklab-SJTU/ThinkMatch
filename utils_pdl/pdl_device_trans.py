import re
import paddle

def place2int( pdl_place):
    device = re.findall(r'\d',str(pdl_place))
    ## Here assmue that a tensor is merely put in one GPU
    return eval(device[0])

def place2str( pdl_place):
    device = re.findall(r'\d',str(pdl_place))
    ## Here assmue that a tensor is merely put in one GPU
    return ("gpu:" + device[0])

