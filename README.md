# Parsing-R-CNN
**(New!)** Official implementation of **Parsing R-CNN for Instance-Level Human Analysis (CVPR 2019)**

## Citing Parsing R-CNN

If you use Parsing R-CNN, please use the following BibTeX entry.

```BibTeX
@inproceedings{yang2019cvpr,
  title = {Parsing R-CNN for Instance-Level Human Analysis},
  author = {Lu Yang and Qing Song and Zhihui Wang and Ming Jiang},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

```

In this repository, we release the Parsing R-CNN code in Pytorch.

- Parsing R-CNN architecture:
<p align="center"><img width="90%" src="data/parsing_rcnn.png" /></p>

- Parsing R-CNN output:
<p align="center"><img width="90%" src="data/output.png" /></p>


## Installation
- 8 x TITAN RTX GPU
- pytorch1.1
- python3.6.8

Install Parsing R-CNN following [INSTALL.md](https://github.com/soeaver/Parsing-R-CNN/blob/master/INSTALL.md#install).


## Results and Models

**On CIHP**



**On MHP-v2**



- 'baseline' denotes our implementation [Parsing R-CNN](https://arxiv.org/abs/1811.12596).
- '+tta' denotes using test-time augmentation, including: soft-nms + bbox voting + h-flipping + multi-scale


**ImageNet pretrained weight**

- [R-50](https://drive.google.com/open?id=1EtqFhrFTdBJNbp67effArVrTNx4q_ELr)
- [R-50-GN](https://drive.google.com/open?id=1EtqFhrFTdBJNbp67effArVrTNx4q_ELr)
- [X-101-32x8d](https://drive.google.com/open?id=1c4OSVZIZtDT49B0DTC0tK3vcRgJpzR9n)


## Training

To train a model with 8 GPUs run:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --cfg cfgs/CIHP/e2e_rp_rcnn_R-50-FPN_3x_ms.yaml
```


## Evaluation

### multi-gpu evaluation,
```
python tools/test_net.py --cfg ckpts/CIHP/e2e_rp_rcnn_R-50-FPN_3x_ms/e2e_rp_rcnn_R-50-FPN_3x_ms.yaml --gpu_id 0,1,2,3,4,5,6,7
```

### single-gpu evaluation,
```
python tools/test_net.py --cfg ckpts/CIHP/e2e_rp_rcnn_R-50-FPN_3x_ms/e2e_rp_rcnn_R-50-FPN_3x_ms.yaml --gpu_id 0
```


## License
Parsing-R-CNN is released under the [MIT license](https://github.com/soeaver/Parsing-R-CNN/blob/master/LICENSE).
