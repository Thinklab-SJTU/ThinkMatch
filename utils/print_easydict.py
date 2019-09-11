from easydict import EasyDict as edict

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(indent_cnt=0)
def print_easydict(inp_dict: edict):
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            print('{}{}:'.format(' ' * 2 * print_easydict.indent_cnt, key))
            print_easydict.indent_cnt += 1
            print_easydict(value)
            print_easydict.indent_cnt -= 1

        else:
            print('{}{}: {}'.format(' ' * 2 * print_easydict.indent_cnt, key, value))

@static_vars(indent_cnt=0)
def print_easydict_str(inp_dict: edict):
    ret_str = ''
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            ret_str += '{}{}:\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key)
            print_easydict_str.indent_cnt += 1
            ret_str += print_easydict_str(value)
            print_easydict_str.indent_cnt -= 1

        else:
            ret_str += '{}{}: {}\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key, value)

    return ret_str
