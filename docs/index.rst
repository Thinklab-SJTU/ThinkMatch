Welcome to ThinkMatch's documentation!
======================================

**ThinkMatch** is a deep graph matching repository developed and maintained by
`ThinkLab <http://thinklab.sjtu.edu.cn>`_ at Shanghai Jiao Tong University.

This repository is developed for the following purposes:

* **Providing modules** for developing deep graph matching algorithms to facilitate future research.
* **Providing implementation** of state-of-the-art deep graph matching methods.
* **Benchmarking** existing deep graph matching algorithms under different dataset & experiment settings, for the purpose of fair comparison.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/introduction
   guide/installation
   guide/experiment
   guide/models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/_autosummary/src.backbone
   api/_autosummary/src.build_graphs
   api/_autosummary/src.factorize_graph_matching
   api/_autosummary/src.displacement_layer
   api/_autosummary/src.evaluation_metric
   api/_autosummary/src.gconv
   api/_autosummary/src.loss_func
   api/_autosummary/src.spectral_clustering
   api/_autosummary/src.lap_solvers
