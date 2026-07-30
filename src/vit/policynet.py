"""CTS 用 PolicyNet：从 training 侧 net.py 按路径加载，避免与 src/policynet 目录的命名空间冲突。"""
import importlib.util
from pathlib import Path

_net_path = Path(__file__).resolve().parent.parent / "policynet" / "policynet" / "net.py"
if not _net_path.is_file():
    raise ImportError(f"PolicyNet implementation not found: {_net_path}")

_spec = importlib.util.spec_from_file_location("srvit_policeynet_net", _net_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
PolicyNet = _mod.PolicyNet

__all__ = ["PolicyNet"]
