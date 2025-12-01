# android_use_tfile
在Android上使用tflite模型进行多类别识别

# 食用教程

## 前言

> 这个模型训练的绝大多代码来自于AI（千问和`Gemini 3 Pro`），我只是收集了数据集，并且`dog`和`cat`的数据集来源于`Kaggle`。其他的数据集来源于百度、Bing的图片搜索。
>
> 注意：该训练使用的是`GPU`训练，因此你应该安装了你的显卡对应（或者支持）的`CUDA`和`cuDNN`。如果你的电脑不支持GPU训练，那么你得让AI生成使用`CPU`训练的脚本。

# 环境

使用的`Python`版本是`3.8.20`，`Python`环境不能小于`3.8.20`。建议创建一个环境以免污染主环境。

创建好并激活环境执行`pip install -r requirements.txt`安装所需的依赖

# 各文件说明

> - `datasets`存放数据集的文件夹
> - `change_file_name.py` 修改`datasets\train_1\xxx`文件夹内的文件名
> - `check_img.py` 检查`datasets\train_1\xxx`文件夹内的图片是否可以用于训练。
> - `split_test_dataset.py` 用于划分`test`数据集的脚本，是使用`predict_test_model.py`的前提。
> - `predict_test_model.py` 当模型训练好以后，用于评估模型
> - `train_main.py` 训练的主程序
> - `models`存放模型的文件夹（可以不用建，脚本会建）
> - `labels`存放标签的文件夹（可以不用建，脚本会建）

# Android

`Android`的版本需要`API > 34` ，创建新项目时，选择 `No Activity` 然后按照`MyAppliction3`补全结构。
