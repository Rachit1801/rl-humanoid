# What is Miniconda?

Miniconda is a minimal installer for the Conda package and environment manager. Conda manages your Python environments and packages. Each project needs different versions of Python and libraries.

The Conda environment is a global virtual environment managed internally by Conda stored hidden away in ~/miniconda3/envs/.

With Conda Each project is completely isolated.

```
Base
│
├── rl-env
│     Python 3.11
│     PyTorch
│     MuJoCo
│
└── data-env
      Python 3.12
      Pandas
      Jupyter
```

| Miniconda                  | Anaconda                          |
| -------------------------- | --------------------------------- |
| ~100 MB                    | ~3–5 GB                           |
| Only Conda                 | Hundreds of packages preinstalled |
| Lightweight ✅              | Heavy                             |
| Install only what you need | Many packages you may never use   |

For pure Python projects, venv is perfectly fine and is part of Python itself. Conda shines when you need packages with compiled dependencies (PyTorch, CUDA toolkits, scientific libraries) or multiple Python versions.

| Feature                     | venv              | Miniconda |
| --------------------------- | ----------------- | --------- |
| Isolated environments       | ✅                 | ✅         |
| Multiple Python versions    | ⚠️ Less convenient | ✅         |
| Install non-Python packages | ❌                 | ✅         |
| Easy ML setup               | ⚠️                 | ✅         |