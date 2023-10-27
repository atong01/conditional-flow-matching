# CIFAR-10 experiments using TorchCFM

This repository is used to reproduce the CIFAR-10 experiments from [1](https://arxiv.org/abs/2302.00482). It is a repository in construction and we will add more features and details in the future (including FID computations and pre-trained weights). We have followed the experimental details provided in [2](https://openreview.net/forum?id=PqvMRDCJT9t).

To reproduce the experiment and save the weights, install the requirements from the main repository and then run (runs on a single RTX 2080 GPU):

```bash
python3 train_cifar10.py --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 800001 --save_step 20000
```

To run a script closer to the original Flow Matching paper, use the following script(might require several GPUs):

```bash
python3 train_cifar10.py --lr 2e-4 --ema_decay 0.9999 --num_channel 256 --batch_size 256 --total_steps 400001 --save_step 20000
```

If you find this code useful in your research, please cite the following papers (expand for BibTeX):

<details>
<summary>
A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023.
</summary>

```bibtex
@article{tong2023improving,
  title={Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author={Tong, Alexander and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and {Rector-Brooks}, Jarrid and Fatras, Kilian and Wolf, Guy and Bengio, Yoshua},
  year={2023},
  journal={arXiv preprint 2302.00482}
}
```

</details>

<details>
<summary>
A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio. Simulation-Free Schrödinger Bridges via Score and Flow Matching, 2023.
</summary>

```bibtex
@article{tong2023simulation,
   title={Simulation-Free Schr{\"o}dinger Bridges via Score and Flow Matching},
   author={Tong, Alexander and Malkin, Nikolay and Fatras, Kilian and Atanackovic, Lazar and Zhang, Yanlei and Huguet, Guillaume and Wolf, Guy and Bengio, Yoshua},
   year={2023},
   journal={arXiv preprint 2307.03672}
}
```
