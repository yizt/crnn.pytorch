# crnn.pytorch

​            本工程使用随机生成的水平和垂直图像训练crnn模型做文字识别;一共使用10多种不同字体;共包括数字、字符、简体和繁体中文字30656个,详见[all_words.txt](data/all_words.txt)。

1. [预测](#预测)<br>
    1.1 [直接预测](#直接预测)<br>
    1.2 [使用restful服务预测](#使用restful服务预测)<br>
2. [模型效果](#模型效果)<br>
    2.1 [水平方向](#水平方向)<br>
    2.2 [垂直方向](#垂直方向)<br>
3. [评估](#评估)<br>
4. [训练](#训练)<br>






## 预测
### 直接预测

   预训练模型下载地址:水平模型 [crnn.horizontal.061.pth](https://pan.baidu.com/s/1beTiTtHMdpOZ9wkqPwODQg) 提取码: fguu; 垂直模型 [crnn.vertical.090.pth](https://pan.baidu.com/s/1VsW2K4G0g0QX5W3Lb3SoAw) 提取码: ygx7。

a) 执行如下命令预测单个图像


```bash
# 水平方向
python demo.py --weight-path /path/to/chk.pth --image-path /path/to/image --direction horizontal
# 垂直方向
python demo.py --weight-path /path/to/chk.pth --image-path /path/to/image --direction vertical

```


b) 执行如下命令预测图像目录

```bash
# 水平方向
python demo.py --weight-path /path/to/chk.pth --image-dir /path/to/image/dir --direction horizontal
# 垂直方向
python demo.py --weight-path /path/to/chk.pth --image-dir /path/to/image/dir --direction vertical
```

### 使用restful服务预测

a) 启动restful服务

```shell
python rest.py -l /path/to/crnn.horizontal.061.pth -v /path/to/crnn.vertical.090.pth -d cuda
```

b) 使用如下代码预测，参考`rest_test.py`

```python
import base64
import requests

img_path = './images/horizontal-002.jpg'

with open(img_path, 'rb') as fp:
    img_bytes = fp.read()

img = base64.b64encode(img_bytes).decode()
data = {'img': img}

r = requests.post("http://localhost:5000/crnn", json=data)

print(r.json()['text'])
```

结果如下：

```shell
 厘鳃 銎 萛闿 檭車 垰銰 陀 婬２ 蠶
```



## 模型效果

​             以下图像均来为生成器随机生成的,也可以试用自己的图像测试

### 水平方向

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


### 垂直方向


| Image1                       | Image2                       | Image3                       | Image4                       | Image5                       | Image6                       | Image7                       | Image8                       | Image9                       | Image10                      |
| ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| ![](images/vertical-001.jpg) | ![](images/vertical-002.jpg) | ![](images/vertical-003.jpg) | ![](images/vertical-004.jpg) | ![](images/vertical-005.jpg) | ![](images/vertical-006.jpg) | ![](images/vertical-007.jpg) | ![](images/vertical-008.jpg) | ![](images/vertical-009.jpg) | ![](images/vertical-010.jpg) |

从左到右识别结果
```
蟒销咔侉糌圻
醵姹探里坌葺神赵漓
紊趼掰膊縉氺月
皱坊凋庳剜蓍赚拾赣缮
ⅲ樱妖遐灌纽枰孽笸逼⊙斟喧湄汇
铳颢汜橇忝稿┗淌㎞琉炭盛㈨事
ゆ囚具憎鉴蔟馍络ら裕翱偬
绸唿綜袼殊潸名廪收鈁跃唤蛴腕
斥嗡门彳鹪Ⅴ戝物据趱欹
覃追煮茫舔酾桎藏瘪挚檎笏嵊疙鹦
```


## 评估

a) 水平方向
```shell
python eval.py --weight-path /path/to/crnn.horizontal.061.pth \
--direction horizontal --device cuda

```
输出结果：`acc:0.893`

b) 垂直方向
```shell
python eval.py --weight-path /path/to/crnn.vertical.090.pth \
--direction vertical --device cuda
```
输出结果：`acc:0.523`

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