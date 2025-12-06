import os
from PIL import Image


def batch_convert_to_jpg(path_name: str):
    # 获取当前工作目录
    current_dir = path_name

    # 定义需要转换的源格式后缀 (忽略已经是 jpg 的文件)
    valid_extensions = ('.webp', '.png', '.bmp', '.tiff', '.tif', '.gif', '.jpg', '.jpeg')

    print(f"正在扫描当前目录: {current_dir} ...")

    # 获取所有文件
    files = [f for f in os.listdir(current_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        print("当前目录下没有找到 PNG, WEBP, BMP 或 TIFF 等图片。")
        return

    count = 0
    for filename in files:
        file_path = os.path.join(current_dir, filename)
        # 生成新的文件名 (例如 image.png -> image.jpg)
        file_name_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(current_dir, f"{file_name_no_ext}.jpg")

        try:
            with Image.open(file_path) as img:
                # 处理 GIF：只转换第一帧
                if img.format == 'GIF':
                    img.seek(0)

                # 核心逻辑：处理透明度 (Alpha通道)
                # 无论是 RGBA 还是 P 模式(索引颜色)，都统一先转为 RGBA 处理
                img = img.convert("RGBA")

                # 创建一个白色背景图
                background = Image.new('RGB', img.size, (255, 255, 255))
                # 将原图粘贴到白底上 (使用 alpha 通道作为蒙版)
                # split()[-1] 取最后一个通道，即 Alpha 通道
                background.paste(img, mask=img.split()[-1])

                # 保存为 JPG
                background.save(output_path, "JPEG", quality=95)

                print(f"[成功] {filename} -> {file_name_no_ext}.jpg")
                count += 1

        except Exception as e:
            print(f"[跳过] {filename} 转换失败: {e}")

    print(f"\n全部完成！共转换了 {count} 张图片。")


if __name__ == "__main__":
    file_name_path = "E:\\tflite_train\\android_use_tflite\\Python\\datasets\\train_1\\car"
    batch_convert_to_jpg(file_name_path)
