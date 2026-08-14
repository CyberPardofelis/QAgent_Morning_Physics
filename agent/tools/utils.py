import tempfile
import subprocess
import yaml
import os
from agent.tools.registry import register_tool


def load_journal_config():
    # 获取当前文件（utils.py）所在目录的父目录（agent/）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'journals.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config.get('journals', {})

JOURNAL_CONFIG = load_journal_config()


@register_tool(name="run_python", is_streaming=False)
def run_python(code: str) -> str:
    """执行 Python 代码（子进程隔离），返回 stdout + stderr 合并输出"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code.encode())
        filename = f.name
    try:
        result = subprocess.run(
            ["python", filename],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,  # 安全：60 秒硬超时
        )
        return (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return "❌ 执行超时（> 60 秒）"
    except Exception as e:
        return f"❌ 执行失败: {e}"
    finally:
        try:
            os.unlink(filename)
        except OSError:
            pass
