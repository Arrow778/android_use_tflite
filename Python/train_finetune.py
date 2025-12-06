import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
import matplotlib.pyplot as plt
import os.path as path
from datetime import datetime
import numpy as np

# ======================
# 1. 全局配置
# ======================
CONFIG = {
    # 路径配置
    "DATASET_PATH": "datasets/train_1",  # 你的数据集路径
    "MODEL_DIR_ROOT": "models",  # 模型保存目录
    "LABEL_DIR_ROOT": "labels",  # 标签保存目录

    # 训练参数
    "IMG_SIZE": (224, 224),
    "BATCH_SIZE": 32,  # 如果显存 > 4GB或者数据量小，建议改为 32
    "EPOCHS": 15,  # 单阶段最大轮数 (配合早停，不用担心过多)
    "LEARNING_RATE": 1e-3,  # 初始学习率 0.001
    "SEED": 100,
    "VAL_RATE": 0.30,  # 验证集比例
}


# ======================
# 2. 工具函数
# ======================
def ensure_dirs_exist():
    """创建必要的文件夹"""
    for d in [CONFIG["MODEL_DIR_ROOT"], CONFIG["LABEL_DIR_ROOT"]]:
        if not path.exists(d):
            os.makedirs(d)
            print(f"📂 Created directory: {d}")


def setup_gpu():
    """配置显存按需增长"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU Ready: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU found. Training will be slow.")


def calculate_class_weights(data_path):
    """计算类别权重，处理样本不平衡"""
    counts = {}
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

    total = 0
    for idx, name in enumerate(class_names):
        p = os.path.join(data_path, name)
        # 只统计图片文件
        c = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        counts[idx] = c
        total += c

    num_classes = len(class_names)
    weights = {}
    if num_classes > 0:
        for idx, count in counts.items():
            if count > 0:
                weights[idx] = (1.0 / count) * (total / num_classes)
            else:
                weights[idx] = 1.0

    print(f"📊 Class Weights calculated. Total images: {total}")
    return weights, num_classes, class_names


def load_datasets(data_path, img_size, batch_size, seed, val_rate):
    """加载数据管线"""
    print("🔄 Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path, validation_split=val_rate, subset="training",
        seed=seed, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path, validation_split=val_rate, subset="validation",
        seed=seed, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=True
    )

    # 性能优化：缓存和预取
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


# ======================
# 3. 模型构建
# ======================
def build_model_graph(num_classes, img_size):
    """
    构建模型：包含数据增强和预训练的 MobileNetV2
    """
    # 1. 强力数据增强 (训练时开启，预测时自动关闭)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.1, 0.1)
    ], name="data_augmentation")

    # 2. 预处理
    preprocess_input = applications.mobilenet_v2.preprocess_input

    # 3. 基础模型 (不含顶层，使用 ImageNet 权重)
    base_model = applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # 🧊 关键：冻结基础模型，只训练新加的层
    base_model.trainable = False

    # 4. 组装
    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = data_augmentation(inputs)
    x = layers.Lambda(preprocess_input)(x)

    # training=False 确保 BN 层使用 ImageNet 的统计数据，而不是当前 Batch 的
    # 这对迁移学习非常重要，能保证稳定性
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # 防止过拟合

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(1e-4)  # 轻微的正则化
    )(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ======================
# 4. 可视化与保存
# ======================
def plot_history(history, save_path):
    """绘制训练曲线"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # 准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # 损失
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(save_path)
    print(f"📈 Training curve saved to {save_path}")


def save_tflite(model, save_path):
    """转换为 TFLite (float16 量化)"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # 半精度量化，减小体积
    tflite_model = converter.convert()

    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"💾 TFLite model saved: {save_path}")


def save_labels(names, save_path):
    """保存标签文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        for n in names:
            f.write(n + '\n')
    print(f"🏷️ Labels saved: {save_path}")


# ======================
# 5. 主程序
# ======================
def main():
    ensure_dirs_exist()
    setup_gpu()

    # 1. 准备数据
    weights, num_classes, class_names = calculate_class_weights(CONFIG["DATASET_PATH"])
    train_ds, val_ds = load_datasets(
        CONFIG["DATASET_PATH"], CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"],
        CONFIG["SEED"], CONFIG["VAL_RATE"]
    )

    # 2. 构建模型
    print("\n🔨 Building Model...")
    model = build_model_graph(num_classes, CONFIG["IMG_SIZE"])
    model.summary()

    # 3. 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. 定义回调函数 (让训练更智能)
    callbacks = [
        # 如果验证集 Loss 3轮不下降，自动减小学习率
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
        ),
        # 如果验证集 Accuracy 6轮不提升，提前结束
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1
        )
    ]

    # 5. 开始训练 (单阶段)
    print(f"\n🚀 Starting Training for {CONFIG['EPOCHS']} epochs...")
    history = model.fit(
        train_ds,
        epochs=CONFIG["EPOCHS"],
        validation_data=val_ds,
        class_weight=weights,
        callbacks=callbacks
    )

    # ==========================
    # 收尾工作
    # ==========================
    timestamp = datetime.now().strftime('%m-%d-%H-%M')

    # 绘图
    plot_history(history, "img\\training_curve_final.png")

    # 保存 TFLite
    # 注意：TFLite Converter 会自动忽略训练专用的层（如 Dropout 和 RandomFlip）
    print("\n📦 Exporting TFLite model...")
    tflite_path = path.join(CONFIG["MODEL_DIR_ROOT"], f"model-final-{timestamp}.tflite")
    save_tflite(model, tflite_path)

    # 保存标签
    label_path = path.join(CONFIG["LABEL_DIR_ROOT"], "label-mutil.txt")
    save_labels(class_names, label_path)

    print(f"\n✅ All Done! 验证集准确率 (Best): {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()