# Tribute to https://github.com/wuyang556/paddlevision
from tqdm import tqdm
from collections import OrderedDict
import paddle.fluid as fluid
from torchvision import models
from paddle import vision


def convert_params(model_th, model_pd, model_path):
    """
    convert pytorch model's parameters into paddlepaddle model, then saving as .pdparams
    :param model_th: pytorch model which has loaded pretrained parameters.
    :param model_pd: paddlepaddle dygraph model
    :param model_path: paddlepaddle dygraph model path
    :return:
    """
    state_dict_th = model_th.state_dict()
    state_dict_pd = model_pd.state_dict()
    state_dict = OrderedDict()
    num_batches_tracked_th = 0
    for key_th in state_dict_th.keys():
        if "num_batches_tracked" in key_th:
            num_batches_tracked_th += 1

    for key_pd in tqdm(state_dict_pd.keys()):
        if key_pd in state_dict_th.keys():
            if "fc" in key_pd or "classifier":
                if len(state_dict_pd[key_pd].shape) < 4:
                    state_dict[key_pd] = state_dict_th[key_pd].numpy().astype("float32").transpose()
                else:
                    state_dict[key_pd] = state_dict_th[key_pd].numpy().astype("float32")
            else:
                state_dict[key_pd] = state_dict_th[key_pd].numpy().astype("float32")
        else:
            if "_mean" in key_pd:
                key_th = key_pd.replace("_mean", "running_mean")
            elif "_variance" in key_pd:
                key_th = key_pd.replace("_variance", "running_var")
            state_dict[key_pd] = state_dict_th[key_th].numpy().astype("float32")
    assert len(state_dict_pd.keys()) == len(state_dict.keys())
    try:
        len(state_dict.keys()) + num_batches_tracked_th == len(state_dict_th.keys())
    except Exception as e:
        print(f"The number of num_batches_tracked is {num_batches_tracked_th} in pytorch model.")
        print("Exception: there are other layer parameter which not converted")

    model_pd.set_dict(state_dict)

    fluid.dygraph.save_dygraph(model_pd.state_dict(), model_path=model_path)
    print("model convert successfully.")


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model_th = models.vgg16_bn(pretrained=True)
        model_pd = vision.models.vgg16(pretrained=False, batch_norm=True)
        model_path = "./vgg16_bn"
        print(model_th.state_dict().keys())
        print(len(model_th.state_dict().keys()))
        print(model_pd.state_dict().keys())
        print(len(model_pd.state_dict().keys()))
        convert_params(model_th, model_pd, model_path)

