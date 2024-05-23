# CLIO

## Getting started

Add this line in your `.zshrc/.bashrc` or other shell-compatible config

```bash
export CLIO=/path/to/this/repository
```

### Dependencies

- [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (Conda also works)
- Latex (for generating PDFs and combining plots)

### Installation

```bash
micromamba env create --name clio --file env.yaml
```

### Activation

```bash
micromamba activate clio
```

### Documents

- [Trace Editor](./docs/trace-editor.md)


### gnuplot

```bash
wget https://psychz.dl.sourceforge.net/project/gnuplot/gnuplot/6.0.0/gnuplot-6.0.0.tar.gz
tar -xvf gnuplot-6*.tar.gz
cd gnuplot-6.0.0
TERMLIBS="-lX11" ./configure --prefix=$HOME/opt
make -j
make check
make install
```

