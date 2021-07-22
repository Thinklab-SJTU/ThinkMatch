Run the Experiment
===================

Run Training and Evaluation
----------------------------

Run training and evaluation
::

    python train_eval.py --cfg path/to/your/yaml

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.
::

    python train_eval.py --cfg experiments/vgg16_pca_voc.yaml

Default configuration files are in ``yaml`` format and are stored in ``experiments/``.

.. note::
    You are welcomed to try your own configurations. If you find a better yaml configuration,
    please let us know by raising an `issue <https://github.com/Thinklab-SJTU/ThinkMatch/issues>`_ or a `PR <https://github.com/Thinklab-SJTU/ThinkMatch/pulls>`_ and we will update the benchmark!

Load Pretrained Models
-----------------------

**ThinkMatch** provides pretrained models. The model weights are available via
`google drive <https://drive.google.com/drive/folders/11xAQlaEsMrRlIVc00nqWrjHf8VOXUxHQ?usp=sharing>`_.

To use the pretrained models, firstly download the weight files, then add the following line to your yaml file:
    ::

        PRETRAINED_PATH: path/to/your/pretrained/weights

The naming of pretrained weights follows:
    ::

        "prefix"_"CNN backbone"_"model name"_"dataset name".pt

Example:
    ::

        pretrained_params_vgg16_bbgm_voc.pt

        "prefix": pretrained_params
        "CNN backbone": vgg16
        "model name": bbgm
        "dataset name": voc
