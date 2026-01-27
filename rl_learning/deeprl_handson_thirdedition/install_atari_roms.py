#!/usr/bin/env python
"""
Atari ROMs 自动安装脚本
"""
import os
import sys
import tarfile
import zipfile
import requests
import subprocess
from pathlib import Path

def download_roms():
    """下载 Atari ROMs"""
    rom_urls = [
        "https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/roms.tar.gz",
        "https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/Roms.rar",
        "https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/ROMS.zip"
    ]
    
    rom_dir = Path("atari_roms")
    rom_dir.mkdir(exist_ok=True)
    
    print("正在下载 Atari ROMs...")
    
    for url in rom_urls:
        try:
            filename = url.split("/")[-1]
            filepath = rom_dir / filename
            
            print(f"尝试从 {url} 下载...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"下载成功: {filename}")
            return filepath
        except Exception as e:
            print(f"下载失败 {url}: {e}")
    
    return None

def extract_roms(filepath):
    """解压 ROMs 文件"""
    print(f"正在解压 {filepath}...")
    
    if filepath.suffix == '.tar.gz':
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(filepath.parent)
        return filepath.parent / "roms"
    
    elif filepath.suffix == '.rar':
        # 需要安装 unrar
        try:
            subprocess.run(["unrar", "x", str(filepath), str(filepath.parent)], 
                          check=True, capture_output=True)
            return filepath.parent
        except:
            print("需要安装 unrar: sudo apt install unrar")
            return None
    
    elif filepath.suffix == '.zip':
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(filepath.parent)
        return filepath.parent
    
    return None

def import_roms(roms_dir):
    """导入 ROMs 到 ale-py"""
    print("正在导入 ROMs 到 ale-py...")
    try:
        import ale_py
        from ale_py.roms.utils import roms_to_import
        
        # 查找 .bin 文件
        bin_files = list(Path(roms_dir).rglob("*.bin"))
        if not bin_files:
            raise FileNotFoundError("未找到 .bin 文件")
        
        # 导入 ROMs
        for rom_path in bin_files:
            try:
                ale_py.ALEInterface().loadROM(str(rom_path))
                print(f"成功导入: {rom_path.name}")
            except:
                pass
        
        # 使用 ale-import-roms 命令
        subprocess.run(["ale-import-roms", str(roms_dir)], 
                      check=True, capture_output=True)
        print("✅ ROMs 导入成功！")
        return True
    except Exception as e:
        print(f"导入失败: {e}")
        return False

def test_atari():
    """测试 Atari 环境"""
    print("\n测试 Atari 环境...")
    try:
        import gymnasium as gym
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        obs, _ = env.reset()
        print(f"✅ 环境创建成功！")
        print(f"   观察空间: {env.observation_space}")
        print(f"   动作空间: {env.action_space}")
        env.close()
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("=" * 50)
    print("Atari ROMs 安装工具")
    print("=" * 50)
    
    # 1. 检查是否已安装必要包
    print("\n1. 检查依赖...")
    try:
        import ale_py
        import gymnasium
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("运行: pip install gymnasium[atari] ale-py")
        return
    
    # 2. 尝试自动导入
    print("\n2. 尝试自动导入 ROMs...")
    try:
        import ale_py
        ale_py.import_roms()
        print("✅ ROMs 自动导入成功！")
    except Exception as e:
        print(f"自动导入失败: {e}")
        
        # 3. 手动下载和导入
        print("\n3. 开始手动下载 ROMs...")
        rom_file = download_roms()
        if not rom_file:
            print("❌ 所有下载源都失败了")
            print("请手动从以下地址下载 ROMs:")
            print("https://github.com/openai/atari-py/tree/master/atari_py/atari_roms")
            return
        
        roms_dir = extract_roms(rom_file)
        if roms_dir and import_roms(roms_dir):
            print("✅ 手动导入成功！")
        else:
            print("❌ 手动导入失败")
    
    # 4. 测试
    print("\n4. 测试环境...")
    if test_atari():
        print("\n🎉 所有设置完成！现在可以运行你的GAN代码了。")
    else:
        print("\n⚠️  环境可能仍有问题，尝试重启终端或使用备选方案。")

if __name__ == "__main__":
    main()