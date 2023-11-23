# Forest-Flow experiment on the Iris dataset using TorchCFM

This notebook is a self-contained example showing how to train the novel Forest-Flow method to generate tabular data [(Jolicoeur-Martineau et al. 2023)](https://arxiv.org/abs/2309.09968). The idea behind Forest-Flow is to **learn Independent Conditional Flow-Matching's vector field with XGBoost models** instead of neural networks. The motivation is that it is known that Forests work currently better than neural networks on Tabular data tasks. This idea comes with some difficulties, for instance how to approximate Flow Matching's loss, and this notebook shows how to do it on a minimal example. The method, its training procedure and the experiments are described in [(Jolicoeur-Martineau et al. 2023)](https://arxiv.org/abs/2309.09968). The full code can be found [here](https://github.com/SamsungSAILMontreal/ForestDiffusion).

To run our jupyter notebooks, installing our package:

```bash
cd ../../

# install torchcfm
pip install -e '.[forest-flow]'

# install ipykernel
conda install -c anaconda ipykernel

# install conda env in jupyter notebook
python -m ipykernel install --user --name=torchcfm

# launch our notebooks with the torchcfm kernel
```
