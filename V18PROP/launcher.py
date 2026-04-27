"""
启动器: 检查依赖并启动天线仿真程序
双击此文件即可运行 (Python 3.6+)
"""
import subprocess
import sys
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(HERE, "antenna_simulator.py")
IMPORTS = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("tkinter", "tkinter"),
]


def show_error(msg):
    """显示错误消息框 (Windows)"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg,
            "天线仿真器 - 依赖检查", 0x10)
    except Exception:
        print(f"\nERROR: {msg}\n")


def main():
    # 检查依赖
    missing = []
    for name, imp in IMPORTS:
        try:
            __import__(imp)
        except ImportError:
            missing.append(name)

    if missing:
        msg = textwrap.dedent(f"""\
            缺少以下 Python 包:
            {', '.join(missing)}

            是否现在自动安装?
            (安装需要联网, 约 2-3 分钟)
        """)
        try:
            import ctypes
            ret = ctypes.windll.user32.MessageBoxW(0, msg,
                "天线仿真器 - 安装依赖", 0x24)  # Yes/No, Warning
            install = (ret == 6)  # IDYES
        except Exception:
            install = True

        if install:
            print("正在安装依赖 (numpy, matplotlib, scipy)...")
            python = sys.executable
            pip = subprocess.run(
                [python, "-m", "pip", "install", "--user",
                 "numpy", "matplotlib", "scipy"],
                capture_output=True, text=True
            )
            if pip.returncode != 0:
                show_error(f"安装失败:\n{pip.stderr[-500:]}")
                return
            print("依赖安装成功!")
        else:
            return

    # 启动主程序
    try:
        python = sys.executable
        subprocess.Popen([python, MAIN_SCRIPT],
                         cwd=HERE)
    except Exception as e:
        show_error(f"启动失败:\n{e}")


if __name__ == "__main__":
    main()
