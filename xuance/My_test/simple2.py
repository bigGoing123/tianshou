import os


def function_a():
    # 假设这是你的相对路径
    relative_path = "./path/to/your/file"

    # 将相对路径传递给函数b
    return relative_path


def function_b(relative_path):
    # 使用os.path.abspath将相对路径转换为绝对路径
    absolute_path = os.path.abspath(relative_path)

    # 输出或返回绝对路径
    print(absolute_path)
