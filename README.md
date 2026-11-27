# MolBridge

MolBridge 根据两种药物的 SMILES 预测相互作用事件类型。输入 CSV 的列名为 `smile_1,smile_2,label`，标签从 0 开始。内置 Deng、Ryu 及 inductive 划分，数据位于根目录的 `data/`。

## 模型如何计算

1. RDKit 将分子转成原子特征和带自环的化学键图。75 维特征依次表示元素、度、隐式价、形式电荷、自由基电子、杂化、芳香性及氢数。每个分子保留全部原子。数据加载器分别打包两列分子，返回 `drug1, drug2, labels`。
2. `model(drug1, drug2)` 接收两个单分子图或两个分子批次，在模型内将化学键图排成两个对角块，再把原子投影到隐藏空间。将每个原子投影成 Q 和 K，各头通过缩放点积和 softmax 得到它对各有效目标原子的边权。建图输入沿用原模型的 `detach()`。取各头均值作为学习图，用一个可学习的全局系数混合化学键图与学习图。
3. 三层 MGRM 共用这张图。每层先做 `LayerNorm(H + A @ H)`，再做 `LayerNorm(Z + FFN(Z))`。将初始表示和各层输出相加，然后对真实原子求和。分类头用 LayerNorm 对每个药对的特征做归一化，尾批只有一个药对也可以训练。

代码按输入到预测的顺序组织。根目录的 `data/` 存放实验数据，`method/data/` 存放处理这些数据的 Python 代码。

| 要改什么 | 对应文件 |
|---|---|
| 75 维原子特征的字段与编码 | [features.py](method/data/features.py) |
| 从 SMILES 构造单分子图、将一列分子打包 | [molecule.py](method/data/molecule.py) |
| 读取药对 CSV、缓存分子、组装数据加载器 | [dataset.py](method/data/dataset.py) |
| 将两种药组成联合图、传播消息并分类 | [model.py](molbridge/model.py) |
| 训练、选模与重评 | [train.py](molbridge/train.py) |
| 计算评价指标、保存逐药对预测 | [metrics.py](molbridge/metrics.py) |

## 安装与运行

需要 Python 3.10 或更新版本。在项目自己的环境中安装适合机器驱动和 GPU 的 PyTorch，再安装其余依赖。使用 RTX 5090 时选用支持 Blackwell 的构建。安装渠道可查阅 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python main.py --dataset Deng --fold 0
```

默认参数在 [config.yml](config.yml)。直接执行 `python main.py` 会训练配置指定的数据集和全部折。可用 `--config path/to/config.yml` 指定配置，配置中的数据及输出目录相对于该文件。按药对在批内动态补齐，显存主要受最长药对影响，可调整 `training.batch_size`。

```bash
python main.py --dataset inductive --fold 0 --device cuda:0
python main.py --evaluate runs/<运行目录>/fold-0/best.pt --device cpu
```

在原运行目录旁重评时，默认读取 `run.json` 记录的数据目录。移动数据后可以用 `--config` 指定新位置。权重旁有 `split_hashes.json` 时，程序会先核验全部测试文件。单独复制的权重也可通过配置指定数据，实际读取的 SHA-256 会写入 `metrics.json`。

## 结果与历史版本

每次训练创建单独的 `runs/` 子目录。每折只用验证集 Macro-F1 选模，同分保留较早轮次。测试读取保存的 `best.pt`。Deng 和 Ryu 评估 test，inductive 分别评估 s1 和 s2。每次独立重评在权重旁新建 eval 目录。

`run.json` 记录配置、实际折号、设备、PyTorch 版本和代码版本。各折目录包含切分哈希、训练曲线 CSV、最优权重、指标和逐样本预测。汇总报告逐折结果及等权平均。Macro-F1 按出现于真实或预测标签中的每个类别分别计算 F1 后取平均。

本版保留原论文的联合建图和 MGRM。实验使用完整分子、LayerNorm 分类头和上述验证集选模规则，从头训练并用本版生成的 `best.pt` 重评。历史配置和实验分支可通过 Git 查看。[重构前的代码](https://github.com/angelfacedac/MolBridge/tree/16d803098cffe7cd205412aecf859e57923a14ce)
