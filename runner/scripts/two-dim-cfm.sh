#!/bin/bash

# Compares flow matching (FM) conditional flow matching (CFM) and optimal
# transport conditional flow matching on four datasets.  twodim is not possible
# for the flow matching algorithm as it has a non-gaussian source distribution.
# FM is therefore only run on three datasets.
python src/train.py -m experiment=cfm \
  model=cfm,otcfm \
  launcher=mila_cpu_cluster \
  model.sigma_min=0.1 \
  datamodule=scurve,moons,twodim,gaussians \
  seed=42,43,44,45,46 &

# Sleep to avoid launching jobs at the same time
sleep 1
python src/train.py -m experiment=cfm \
  model=fm \
  launcher=mila_cpu_cluster \
  model.sigma_min=0.1 \
  datamodule=scurve,moons,gaussians \
  seed=42,43,44,45,46 &
