import tempfile
import subprocess
import yaml

def load_journal_config(config_path='./agent/journals.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config.get('journals', {})

JOURNAL_CONFIG = load_journal_config()

def run_python(code: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code.encode())
        filename = f.name
    result = subprocess.run(
        ["python", filename],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return (result.stdout or "") + (result.stderr or "")