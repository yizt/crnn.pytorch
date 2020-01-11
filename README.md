# crnn.pytorch

​            本工程使用随机生成的水平和垂直图像训练crnn模型做文字识别;一共使用10多种不同字体;共包括数字、字符、简体和繁体中文字30656个,详见[all_words.txt](data/all_words.txt)。


## 预测
   预训练模型下载地址:水平模型 [crnn.horizontal.060.pth]( https://pan.baidu.com/s/1NxR6XwJgPx9kslbFMO0X0A) 提取码: k92d; 垂直模型 [crnn.vertical.090.pth](https://pan.baidu.com/s/1VsW2K4G0g0QX5W3Lb3SoAw) 提取码: ygx7。

a) 执行如下命令预测单个图像

```bash
python demo.py --weight-path /path/to/chk.pth --image-path /path/to/image
```

b) 执行如下命令预测图像目录

```bash
python demo.py --weight-path /path/to/chk.pth --image-dir /path/to/image/dir
```





## 模型效果

​             以下图像均来为生成器随机生成的,也可以试用自己的图像测试

| 图像 | 识别结果 |
| ------------------------------ | ---------------------------- |
| ![](images/horizontal-001.jpg) | 鎏贬冱剽粥碍辆置钷固闻塔ど船 |
| ![](images/horizontal-002.jpg) | 厘鳃銎萛闿檭車垰銰陀婬２蠶 |
| ![](images/horizontal-003.jpg) | 磨丢河窜蹬奶鼋 |
| ![](images/horizontal-004.jpg) | 添肃琉恪范粼兢俺斋┟傺怃梗纱脉陷荼荡荫驿 |
| ![](images/horizontal-005.jpg) | 荼反霎吕娟斑恃畀貅引铥哳断替碱嘏 |
| ![](images/horizontal-006.jpg) | 汨鑅譜軥嶰細挓 |
| ![](images/horizontal-007.jpg) | 讵居世鄄钷橄鸠乩嗓犷魄芈丝 |
| ![](images/horizontal-008.jpg) | 憎豼蕖蚷願巇廾尖瞚寣眗媝页锧荰瞿睔 |
| ![](images/horizontal-009.jpg) | 休衷餐郄俐徂煅黢让咣 |
| ![](images/horizontal-010.jpg) | 桃顸噢伯臣 |
| ![](images/vertical-001.jpg) | 蟒销咔侉糌圻 |
| ![](images/vertical-002.jpg) | 醵姹探里坌葺神赵漓 |
| ![](images/vertical-003.jpg) | 紊趼掰膊縉氺月 |
| ![](images/vertical-004.jpg) | 皱坊凋庳剜蓍赚拾赣缮 |
| ![](images/vertical-005.jpg) | ⅲ樱妖遐灌纽枰孽笸逼⊙斟喧湄汇 |
| ![](images/vertical-006.jpg) | 铳颢汜橇忝稿┗淌㎞琉炭盛㈨事 |
| ![](images/vertical-007.jpg) | ゆ囚具憎鉴蔟馍络ら裕翱偬 |
| ![](images/vertical-008.jpg) | 绸唿綜袼殊潸名廪收鈁跃唤蛴腕 |
| ![](images/vertical-009.jpg) | 斥嗡门彳鹪Ⅴ戝物据趱欹 |
| ![](images/vertical-010.jpg) | 覃追煮茫舔酾桎藏瘪挚檎笏嵊疙鹦 |



## 训练

a) 单机多卡
```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1,2,3,4
python -m torch.distributed.launch --nproc_per_node 4 train.py --device cuda --direction vertical
```

b) 多机多卡
```shell
# 第一台主机
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node 3 --nnodes=2 --node_rank=0 \
--master_port=6066 --master_addr="192.168.0.1" \
train.py --device cuda --direction vertical 

# 第二台主机
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node 3 --nnodes=2 --node_rank=1 \
--master_port=6066 --master_addr="192.168.0.1" \
train.py --device cuda --direction vertical 
```
存在问题:多机训练比单机要慢很多,目前尚未解决.