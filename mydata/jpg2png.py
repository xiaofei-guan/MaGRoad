import os
import sys
from PIL import Image
from multiprocessing import cpu_count, Pool


def convert_image(args):
    """
    使用Pillow进行快速图像转换的辅助函数
    """
    input_path, output_path = args
    try:
        with Image.open(input_path) as img:
            # 禁用渐进式加载以加快读取速度
            img.load()
            # 快速保存参数：关闭优化，压缩级别1
            img.save(output_path, 'PNG', compress_level=1, optimize=False)
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


def batch_convert(input_path, output_dir=None):
    """
    主转换函数
    """
    # 验证输入文件
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found")

    if not input_path.lower().endswith(('.jpg', '.jpeg')):
        raise ValueError("Input file must be a JPEG image")

    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.png")

    # 使用多核处理
    pool = Pool(processes=cpu_count())
    result = pool.map(convert_image, [(input_path, output_path)])
    pool.close()
    pool.join()

    if all(result):
        print(f"Successfully converted to {output_path}")
        return output_path
    else:
        raise RuntimeError("Conversion failed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python converter.py <input.jpg> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        batch_convert(input_file, output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)