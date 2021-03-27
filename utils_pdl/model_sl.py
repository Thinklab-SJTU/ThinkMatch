import paddle
#from paddle.nn import DataParallel


def save_model(model, path):
    """
    if isinstance(model, DataParallel):
        model = model.module
    """

    paddle.save(model.state_dict(), path)


def load_model(model, path):
    """
    if isinstance(model, DataParallel):
        model.module.load_state_dict(paddle.load(path))
    else:
    """
    model.set_state_dict(paddle.load(path))
