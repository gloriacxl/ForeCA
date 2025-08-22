# Leveraging Foreground Collaboration and Augmentation for Industrial Anomaly Detection
This is the code for paper : Leveraging Foreground Collaboration and Augmentation for Industrial Anomaly Detection [IEEE Sensors Journal 24']

~~Complete codes will be available upon acceptance:)~~

# Datasets
* [VisA dataset](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_23)
* [MPDD dataset](https://ieeexplore.ieee.org/abstract/document/9631567)

# Pretrained Models
We provide our model checkpoints to reproduce the performance report in the papar at : [Baidu Drive](https://pan.baidu.com/s/197I3k0q4FUchIrIxd9ABfQ) (password:anqu)

# Experimental Results
![image](https://github.com/gloriacxl/ForeCA/blob/main/figs/experimentalresults.PNG)

# Visualization
![image](https://github.com/gloriacxl/ForeCA/blob/main/figs/visualization.png)

===================================UPDATE===================================
# Evaluating
You need to set **data_path** to the directory where your dataset is stored, and place the downloaded checkpoint in **checkpoint_path**.
```python
python test_ForeCA.py
```

or you can use the shell script,
```python
bash test_ForeCA.sh
```

# Training
If you want to train a model from scratch, the train script is as follows,
```python
python train_ForeCA.py
```
Here, **data_path** refers to the directory that contains your dataset along with its foreground masks. You can obtain it from [Google Drive](https://drive.google.com/drive/folders/1B1ryBLPw9VLDzI3vxokaImGcK5jXoHlj?usp=sharing), or generate it yourself using other segmentation models.

# Citation
If you find this project helpful for your research, please consider citing this paper:
```bibtex
@article{chen2024leveraging,
  title={Leveraging Foreground Collaboration and Augmentation for Industrial Anomaly Detection},
  author={Chen, Xiaolu and Xu, Haote and Wang, Jiaxiang and Tu, Xiaotong and Ding, Xinghao and Huang, Yue},
  journal={IEEE Sensors Journal},
  volume={24},
  number={19},
  pages={30706--30714},
  year={2024},
  publisher={IEEE}
}
