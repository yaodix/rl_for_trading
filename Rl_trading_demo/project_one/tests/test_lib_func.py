import unittest
from pathlib import Path

from pathlib import Path

# 基础创建
p = Path("folder/file.txt")              # 相对路径
p = Path("/home/user/folder/file.txt")    # 绝对路径 (Linux/Mac)
p = Path("C:/Users/user/folder/file.txt") # 绝对路径 (Windows)
p = Path()                                 # 当前目录
print(f"Path()={p}")
p = Path.home()                            # 用户主目录
print(f"Path.home()={p}")
p = Path.cwd()                              # 当前工作目录
print(f"Path.cwd()={p}")


# 拼接路径
p = Path("folder") / "subfolder" / "file.txt"
print(f"Path('folder') / 'subfolder' / 'file.txt'={p}")
p = Path("folder").joinpath("subfolder", "file.txt")
print(f"Path('folder').joinpath('subfolder', 'file.txt')={p}")
