# Development

Create an isolated environment and install the editable package:

```bash
python -m pip install -e ".[all,dev]"
pytest -q
ruff check src tests examples
```

The unit suite uses analytic toy magnification evaluators and does not spend
VBM calls. The documented figure is the lightweight direct-VBM integration
test:

```bash
python examples/build_and_plot.py
```

Before changing numerical defaults, benchmark at least one smooth event, one
central-caustic crossing, and one planetary-caustic event. Report cold JAX
compile time separately from steady-state scan time. Any new error claim must
state its reference function and whether radial interpolation and VBM internal
tolerances are enclosed.

