#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pysair",
    version="0.1",
    author="Rıdvan Örsvuran",
    author_email="orsvuran@geoazur.unice.fr",
    description="simple tools for adjoint inversions",
    license="GPLv3+",
    keywords="adjoint tomography seismology",
    url="http://rdno.org",
    include_package_data=True,
    packages=find_packages(),
    long_description=read('README.md'),
    entry_points={
        "console_scripts": [
            "sair-run=sair.runner:main",
            "sair-run-multiple=sair.runner:multiple_run_command",
            "sair-prepare-models=sair.runner:prepare_models",
            "sair-compute-windows=sair.windows:main",
            "sair-adjoint=sair.adjoint:main",
            "sair-adjoint-su=sair.adjoint_su:main",
            "sair-sum-kernels=sair.sum_kernels:main",
            "sair-precondition=sair.preconditioner:main",
            "sair-smooth-kernels=sair.smooth_kernels:main",
            "sair-clip-kernels=sair.clip_kernels:main",
            "sair-mask-kernels=sair.mask_kernels:main",
            "sair-smooth-around-sources=sair.smooth_around_sources:main",
            "sair-find-dd-pairs=sair.dd_pair:main",
            "sair-generate-update=sair.generate_update:main",
            "sair-update-model=sair.update_model:main",
            "sair-model-misfit=sair.model_misfit:main",
            "sair-plot-model=sair.plot.model:main",
            "sair-plot-kernel=sair.plot.kernel:main",
            "sair-plot-improvement=sair.plot.improvement:main",
            "sair-weights-calc-rec-weights=sair.weights:calc_rec_weights",
            "sair-weights-plot-rec-weights=sair.weights:plot_rec_weights",
            "sair-weights-plot-dd-weights=sair.weights:plot_dd_weights",
            "sair-weights-calc-src-weights=sair.weights:calc_src_weights",
            "sair-weights-plot-src-weights=sair.weights:plot_src_weights",
            "sair-explore=sair.explorer:explore"
        ]
    }
)
