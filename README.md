# crnn.pytorch

​            本工程使用随机生成的水平和垂直图像训练crnn模型做文字识别;一共使用10多种不同字体;共包括数字、字符、简体和繁体中文字30656个,详见[all_words.txt](data/all_words.txt)。


## 预测
   预训练模型下载地址: ![crnn.vertical.090.pth](https://pan.baidu.com/s/1VsW2K4G0g0QX5W3Lb3SoAw) 提取码: ygx7 

```bash
python demo.py --weight-path /path/to/chk.pth --image-path /path/to/image
```



## 模型效果

​             以下图像均来为生成器随机生成的,也可以试用自己的图像测试

|图像|识别结果|
|------------------------------|----------------------------|
|![](images/horizontal-001.jpg)|啊|
|![](images/horizontal-002.jpg)||
|![](images/horizontal-003.jpg)||
|![](images/horizontal-004.jpg)||
|![](images/horizontal-005.jpg)||
|![](images/horizontal-006.jpg)||
|![](images/horizontal-007.jpg)||
|![](images/horizontal-008.jpg)||
|![](images/horizontal-009.jpg)||
|![](images/horizontal-010.jpg)||
|![](images/vertical-001.jpg)|蟒销咔侉糌圻|
|![](images/vertical-002.jpg)|醵姹探里坌葺神赵漓|
|![](images/vertical-003.jpg)|紊趼掰膊縉氺月|
|![](images/vertical-004.jpg)|皱坊凋庳剜蓍赚拾赣缮|
|![](images/vertical-005.jpg)|ⅲ樱妖遐灌纽枰孽笸逼⊙斟喧湄汇|
|![](images/vertical-006.jpg)|铳颢汜橇忝稿┗淌㎞琉炭盛㈨事|
|![](images/vertical-007.jpg)|ゆ囚具憎鉴蔟馍络ら裕翱偬|
|![](images/vertical-008.jpg)|绸唿綜袼殊潸名廪收鈁跃唤蛴腕|
|![](images/vertical-009.jpg)|斥嗡门彳鹪Ⅴ戝物据趱欹|
|![](images/vertical-010.jpg)|覃追煮茫舔酾桎藏瘪挚檎笏嵊疙鹦|



## 训练

a) 单机多卡
```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1,2,3,4
python -m torch.distributed.launch --nproc_per_node 4 train.py --device cuda --direction vertical
```

b) 多机多卡
```shell
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node 3 --nnodes=2 --node_rank=0 \
--master_port=6066 --master_addr="192.168.0.1" \
train.py --device cuda --direction vertical 


export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch --nproc_per_node 3 --nnodes=2 --node_rank=1 \
--master_port=6066 --master_addr="192.168.0.1" \
train.py --device cuda --direction vertical 
```
存在问题:多机训练比单机要慢很多,目前尚未解决.