<font face="monospace">

<center>

# my-transformer

</center>

自然语言处理课程大作业：Attention Is All You Need 复现

### `声明`

本项目使用 MIT 协议开源，任何人可以自由使用、修改、分发本项目的代码。
使用、修改、分发本项目的代码时，请注明出处。如有任何不良后果，作者不承担任何责任。

本项目是论文《Attention Is All You Need》的 PyTorch 实现，
但不保证完全符合论文中的公式和结构。

## 介绍

本项目的目标是复现论文《Attention Is All You Need》[^2]，
但使用与原论文不同的数据集，并进行相应的实验。

数据集来源于课程上的参考实验[^1]中的 exp8，进行英语到法语的翻译任务。
本项目将数据集也附在仓库当中，并已经进行了预处理。

训练使用 train.py，测试使用 test.py，推理使用 infer.py。

<center>

### 参考文献

</center>

[^1]: Rao D, McMahan B. Natural language processing with PyTorch: build intelligent language applications using deep learning[M]. " O'Reilly Media, Inc.", 2019.
[^2]: Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.
