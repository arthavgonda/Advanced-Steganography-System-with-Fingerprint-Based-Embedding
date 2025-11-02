import os

def normalize_path(path: str) -> str:
    if not path:
        return path
    path = path.strip()
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]
    return os.path.expanduser(path)