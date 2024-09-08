# GET-UP
Pytorch implementation of GET-UP: GEomeTric-aware Depth Estimation with Radar Points UPsampling (Accepted by WACV 2025)

Paper link: https://arxiv.org/abs/2409.02720

Models have been tested using Python 3.7/3.8, Pytorch 1.10.1+cu111

## Setting up dataset
To set up the dataset, please refer to the [CaFNet repo](https://github.com/harborsarah/CaFNet).

## Training GET-UP
To train GET-UP on the nuScenes dataset, you may run:
```
python main.py arguments_train_nuscenes.txt
```

## Evaluating GET-UP
To evaluate GET-UP on the nuScenes dataset, you may run:
```
python main_test.py arguments_test_nuscenes.txt
```
You may replace the path dirs in the arguments files.

## Acknowledgement
Our work builds on and uses code from [radar-camera-fusion-depth](https://github.com/nesl/radar-camera-fusion-depth), [bts](https://github.com/cleinc/bts). We'd like to thank the authors for making these libraries and frameworks available.

## Citation
If you use this work, please cite our paper:

```
@misc{getup,
      title={GET-UP: GEomeTric-aware Depth Estimation with Radar Points UPsampling}, 
      author={Huawei Sun and Zixu Wang and Hao Feng and Julius Ott and Lorenzo Servadei and Robert Wille},
      year={2024},
      eprint={2409.02720},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.02720}, 
}
``` 
