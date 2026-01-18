import importlib
import sys

mods = [
    'torch',
    'torch_geometric',
    'torch_scatter',
    'torch_sparse',
    'ignite',
    'jarvis',
    'pydantic',
    'sklearn',
    'matplotlib',
    'sympy'
]

print('Python:', sys.version)

for m in mods:
    try:
        importlib.import_module(m)
        print(f'{m}: OK')
    except Exception as e:
        print(f'{m}: MISSING/ERROR: {e}')

try:
    import torch
    print('torch version:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
except Exception as e:
    print('torch info error:', e)