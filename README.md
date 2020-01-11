# crnn.pytorch



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