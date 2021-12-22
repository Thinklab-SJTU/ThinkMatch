===============================
Introduction to Graph Matching
===============================

Graph Matching (GM) is a fundamental yet challenging problem in computer vision, pattern recognition and data mining. GM aims to find node-to-node correspondence among multiple graphs, by solving an NP-hard combinatorial problem named Quadratic Assignment Problem (QAP). Recently, there is growing interest in developing deep learning based graph matching methods.

Graph matching techniques have been applied to the following applications:

* `Bridging movie and synopses <https://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_A_Graph-Based_Framework_to_Bridge_Movies_and_Synopses_ICCV_2019_paper.pdf>`_

  .. image:: ../images/movie_synopses.png

* `Image correspondence <https://arxiv.org/pdf/1911.11763.pdf>`_

  .. image:: ../images/superglue.png

* `Molecules matching <https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Combinatorial_Learning_of_Graph_Edit_Distance_via_Dynamic_Embedding_CVPR_2021_paper.pdf>`_

  .. image:: ../images/molecules.png

* and more...

In this repository, we mainly focus on image keypoint matching because it is a popular testbed for existing graph matching methods.

Readers are referred to the following surveys for more technical details about graph matching:

* Junchi Yan, Shuang Yang, Edwin Hancock. "Learning Graph Matching and Related Combinatorial Optimization Problems." *IJCAI 2020*.
* Junchi Yan, Xu-Cheng Yin, Weiyao Lin, Cheng Deng, Hongyuan Zha, Xiaokang Yang. "A Short Survey of Recent Advances in Graph Matching." *ICMR 2016*.
