import os


def batch_rename(folder_path, new_prefix, overwrite=False):
    """
    批量重命名图片文件

    :param folder_path: 图片所在的文件夹路径
    :param new_prefix: 新文件名前缀 (例如 "cat", "eyeglasses-")
    :param overwrite: 是否覆盖已存在的目标文件（默认 False → 跳过）
    """

    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到文件夹 {folder_path}")
        return

    # 支持的图片扩展名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    files = os.listdir(folder_path)
    images = [f for f in files if f.lower().endswith(valid_extensions)]

    # 尝试按数字排序（适用于 1.jpg, 2.jpg...）
    try:
        images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except (ValueError, TypeError):
        images.sort()  # 回退到字母序

    print(f"📂 正在处理文件夹: {folder_path}")
    print(f"🔢 共找到 {len(images)} 张图片，准备重命名为 {new_prefix}1.jpg ~ {new_prefix}{len(images)}.jpg")

    if not images:
        print("⚠️ 文件夹中没有图片，退出。")
        return

    confirm = input("⚠️ 是否继续？(y/n): ")
    if confirm.lower() != 'y':
        print("已取消。")
        return

    count = 0
    skipped = 0
    for index, filename in enumerate(images, start=1):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        new_filename = f"{new_prefix}{index}{ext}"
        new_path = os.path.join(folder_path, new_filename)

        # 如果新旧路径相同，跳过
        if old_path == new_path:
            continue

        # 检查目标文件是否已存在
        if os.path.exists(new_path):
            if overwrite:
                try:
                    os.remove(new_path)  # 先删除旧文件
                    os.rename(old_path, new_path)
                    count += 1
                    continue
                except Exception as e:
                    print(f"❌ 覆盖并重命名 {filename} 失败: {e}")
                    continue
            else:
                print(f"⚠️ 跳过 {filename} → 目标文件 {new_filename} 已存在")
                skipped += 1
                continue

        # 执行重命名
        try:
            os.rename(old_path, new_path)
            count += 1
            if count % 50 == 0:
                print(f"✅ 已成功重命名 {count} 张...")
        except Exception as e:
            print(f"❌ 重命名 {filename} 失败: {e}")

    print(f"\n🎉 完成！成功重命名 {count} 张，跳过 {skipped} 张（因目标文件已存在）。")


# ==========================================
# 👇 配置区
# ==========================================

if __name__ == "__main__":
    file_name = "car"
    target_folder = f"datasets/train_1/{file_name}"  # 图片文件夹
    prefix_name = f"{file_name}_"  # 前缀

    # ⚠️ 设置 overwrite=True 会覆盖已存在的同名文件（慎用！）
    batch_rename(target_folder, prefix_name, overwrite=False)