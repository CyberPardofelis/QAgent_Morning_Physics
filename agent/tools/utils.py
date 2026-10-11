import contextlib
import os
import subprocess
import sys
import tempfile

import yaml

# 2026-09-04 用户拍板:暂时从 LLM 工具列表中移除 run_python(安全考虑)
# 函数保留 — 后续如果要加回(加 sandbox/资源限制),可以恢复
# from agent.tools.registry import register_tool


def load_journal_config():
    # 获取当前文件（utils.py）所在目录的父目录（agent/）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'journals.yaml')
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config.get('journals', {})

JOURNAL_CONFIG = load_journal_config()


# 2026-09-04 注:run_python 工具已停用(用户拍板),函数暂时保留以备未来加回
# (加 sandbox/资源限制/白名单 import 后再加),目前是 dead code
# @register_tool(name="run_python", is_streaming=False, cost_tier="L0")
def run_python(code: str) -> str:
    """执行 Python 代码(子进程隔离),返回 stdout + stderr 合并输出。
    **当前未注册为 LLM 工具**(2026-09-04 起),仅供内部调用或未来加回。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code.encode())
        filename = f.name
    try:
        # ★ 就地设 UTF-8, 不 import agent.tools.research._proc ★
        # 那条路会拖进整个 research 包(20+ 模块, 含全部 research 工具注册),
        # 为一个 dead code 函数付这个代价不值。规则本身在 _proc.py 里有文档。
        # 不设的后果: 子进程按 cp936 写字节 -> 这里按 utf-8 解 -> 乱码,
        # 而**返回的是一坨乱码字符串, 不抛异常**, 调用方看不出出事了。
        _env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        result = subprocess.run(
            [sys.executable, filename],   # 不写 "python": 可能是另一个解释器
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_env,
            timeout=60,  # 安全:60 秒硬超时
        )
        return (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return "❌ 执行超时(> 60 秒)"
    except Exception as e:
        return f"❌ 执行失败: {e}"
    finally:
        with contextlib.suppress(OSError):
            os.unlink(filename)
