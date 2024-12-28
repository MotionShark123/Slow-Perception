<h3><a href="">Slow Perception:Let's Perceive Geometric Figures Step-by-step</a></h3>
<a href="https://drive.google.com/drive/folders/16N6ptKENnyvAuJq7ZF6BtMWiqUsNTobc?usp=sharing"><img src="https://img.shields.io/badge/data-yellow"></a>
<a href="https://drive.google.com/drive/folders/16N6ptKENnyvAuJq7ZF6BtMWiqUsNTobc?usp=sharing"><img src="https://img.shields.io/badge/weights-red"></a>
<a href="https://github.com/Ucas-HaoranWei/Slow-Perception/blob/main/Slow_perception.pdf"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 

[Haoran Wei*](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), Youyang Yin*, Yumeng Li, Jia Wang, [Liang Zhao](https://scholar.google.com.hk/citations?user=uJJ5zskAAAAJ&hl=zh-CN&oi=sra),  [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), [Zheng Ge](https://joker316701882.github.io/), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en)

<p align="left">
<img src="assets/img1.jpg" style="width: 255px" align=left>
<img src="assets/geometric.gif" style="width: 200px" align=center>
</p>



&emsp;&emsp;&emsp;&emsp; ***Accurate copying is the first step to visual o1!***




## Release
- [2024/12/24]🔥🔥🔥 We release the slow perception! The paper can be found [here](https://github.com/Ucas-HaoranWei/Slow-Perception/blob/main/Slow_perception.pdf) temporarily and we will submit it to arxiv after we completing the appendix part.



[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)



## Contents
- [Install](#install)
- [Weights](#weights)
- [Data&Benchmark](#data-prepare)
- [Eval](#eval)
- [Train](#train)



## Install
0. The codebase is based on [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0), and if you have installed the GOT environment, use the GOT conda is OK.
1. Clone this repository and navigate to the Slow-Perception-master folder
```bash
git clone https://github.com/Ucas-HaoranWei/Slow-Perception.git
cd 'Slow-Perception-master'
```
2. Install Package
```Shell
conda create -n sp python=3.10 -y
conda activate sp
pip install -e .
```

3. Install Flash-Attention
```
pip install ninja
pip install flash-attn --no-build-isolation
```

## Weights
- [Google Drive](https://drive.google.com/drive/folders/16N6ptKENnyvAuJq7ZF6BtMWiqUsNTobc?usp=sharing)
1. Download the SP-1/weights.zip to Slow-Perception-master
```Shell
unzip weights.zip
```   
2. We provide the baseline and 4-length perceptual ruler weights.


## Data-prepare
- [Google Drive](https://drive.google.com/drive/folders/16N6ptKENnyvAuJq7ZF6BtMWiqUsNTobc?usp=sharing)
1. Download the SP-1/train_sp1.zip and all SP-1/*.json to Slow-Perception-master for train
```Shell
unzip train_sp1.zip
```
2. Download the SP-1/benchmarks.zip  to Slow-Perception-master for eval.
```Shell
unzip benchmarks.zip
```
**Note**:
The folders hierarchy are as follows:
```
  --Slow-Perception-master
      --SP-1
      --SP  
      --...
```

## Eval

```Shell
python3 SP/demo/run_jihe_parsing.py  --model-name SP-1/weights/4ruler/  --image-file SP-1/benchmarks/val_set/
```
```Shell
python3 calculate_f1.py
```
If you want to input a single image:
```Shell
python3 SP/demo/run_jihe_parsing.py  --model-name SP-1/weights/4ruler/  --image-file results/jihe_demo.jpg
```

## Train
1. Download the [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) weights .
```Shell
deepspeed     SP/train/train_SP.py \
 --deepspeed   zero_config/zero2.json \
 --model_name_or_path /GOT_weights/   \
 --freeze_vision_tower False \
 --freeze_lm_model False  \
 --vision_select_layer -2 \
 --use_im_start_end True   \
 --fp16 True   \
 --gradient_accumulation_steps 2    \
 --evaluation_strategy "no"   \
 --save_strategy "steps"  \
 --save_steps 1000   \
 --save_total_limit 1   \
 --weight_decay 0.    \
 --warmup_ratio 0.003     \
 --lr_scheduler_type "cosine"    \
 --logging_steps 1    \
 --tf32 True     \
 --model_max_length 4096    \
 --gradient_checkpointing True   \
 --dataloader_num_workers 8    \
 --report_to none  \
 --per_device_train_batch_size 2    \
 --num_train_epochs 2  \
 --learning_rate 3e-5   \
 --datasets  SP-1 \
 --output_dir jihe_sp_4ruler/ \
```

## Contact
Don't hesitate to contact me by email, weihaoran18@mails.ucas.ac.cn, if you have any questions.


## Acknowledgement
- [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0): the codebase we built upon!

## Citation
Coming soon!


