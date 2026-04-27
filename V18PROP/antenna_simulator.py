"""
1420 MHz 天线仿真器 v2.1 — 入口
"""
import os
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from gui import App


def _setup_font():
    """配置中文字体"""
    for fn in ['Microsoft YaHei', 'SimHei', 'DengXian',
               'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']:
        try:
            plt.rcParams['font.family'] = fn
            f, a_ = plt.subplots(figsize=(0.01, 0.01))
            a_.set_title("测试")
            plt.close(f)
            break
        except Exception:
            plt.rcParams['font.family'] = 'sans-serif'
            continue
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False


def main():
    import traceback
    try:
        _setup_font()
        root = tk.Tk()
        App(root)
        root.mainloop()
    except Exception as e:
        lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_log.txt')
        with open(lp, 'w', encoding='utf-8') as f:
            f.write(f"错误: {e}\n\n")
            traceback.print_exc(file=f)
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"程序启动失败:\n\n{e}\n\n详情: {lp}",
                "天线仿真器 - 错误", 0x10)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
