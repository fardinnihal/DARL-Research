# DARL study — Stage 1: DARL module and unit tests (validation fixes applied)

Stage 1 contains only the DARL activation (proposal §5.1–5.3) and its tests.
DSReLU, ReDo and the training pipeline come after these tests pass on your machine.

## Project structure

```
darl-study/
├── pyproject.toml                    # package metadata, dependencies, pytest settings
├── README.md                         # this file
├── .gitignore
├── src/
│   └── darl_study/
│       ├── __init__.py
│       └── activations/
│           ├── __init__.py           # exports DARL, darl_slope, darl_tracking, inactive_fraction
│           └── darl.py               # the DARL nn.Module
├── scripts/
│   └── darl_demo.py                  # prints how D and alpha evolve for three units
└── tests/
    ├── test_darl.py                  # 55 original unit tests (54 run on CPU, 1 needs CUDA)
    ├── test_darl_validation.py       # 35 regression tests for the input/state validation fixes
    └── test_darl_integration.py      # 16 tests: real optimizer steps and full checkpoint resume
```

## Install (Windows 11, CPU, PyCharm)

Requires Python 3.10 or newer.

1. Unzip `darl-study` and open the folder in PyCharm (**File → Open**).
2. Open PyCharm's **Terminal** tab (it starts in the project folder) and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` from a `cmd` terminal.

3. In PyCharm, set the interpreter to the new environment:
   **Settings → Project → Python Interpreter → Add Interpreter → Existing → `.venv\Scripts\python.exe`**.
4. Set the test runner: **Settings → Tools → Python Integrated Tools → Default test runner: pytest**.

## Run

From PyCharm's **Terminal** tab, with `.venv` active, in the `darl-study` folder:

```powershell
# every test (all three files)
python -m pytest -v

# one file at a time
python -m pytest -v tests\test_darl.py
python -m pytest -v tests\test_darl_validation.py
python -m pytest -v tests\test_darl_integration.py

# demo
python scripts\darl_demo.py
```

In the PyCharm GUI: right-click the `tests` folder → **Run 'pytest in tests'** to run everything, or right-click a single test file → **Run 'pytest in <file>'**.

If you installed an earlier Stage 1 copy, run `pip install -e ".[dev]"` again in the new folder so Python imports this version.

## Expected output and pass criteria

**Tests:** the last line should read

```
105 passed, 1 skipped in X.XXs
```

Per file: `test_darl.py` 54 passed, 1 skipped; `test_darl_validation.py` 35 passed; `test_darl_integration.py` 16 passed.

The one skipped test is `test_cuda_matches_cpu`, which runs only where CUDA exists (it will run on Colab). **Any failure or error means Stage 1 is not complete.** Send the full `pytest -v` output.

**Demo:** the values are fixed by the equations, so they should match these exactly to 4 decimals:

```
 step |     D_0     D_1     D_2 |     a_0     a_1     a_2
    1 |  0.0100  0.0000  0.0050 |  0.0129  0.0100  0.0115
   10 |  0.0956  0.0000  0.0478 |  0.0377  0.0100  0.0239
  100 |  0.6340  0.0000  0.3170 |  0.1939  0.0100  0.1019
  500 |  0.9934  0.0000  0.4967 |  0.2981  0.0100  0.1540
 1000 |  1.0000  0.0000  0.5000 |  0.3000  0.0100  0.1550
eval pass changed state: False
```

Check: unit 0 (always inactive) satisfies D = 1 − 0.99^t, so 0.6340 at t = 100. Unit 2 (inactive half the time) settles at D = 0.5, α = 0.155.

## What the tests check

| Group | What is verified | Proposal / roadmap reference |
|---|---|---|
| Output | x for x ≥ 0, α_j·x for x < 0, per-unit and per-channel α, input validation | §5.1, D3 |
| EMA deadness | r_j counts z ≤ 0 (including 0 and −0.0); single update, closed-form multi-step sequence, ρ = 0, conv pooling; α at step t uses D(t); no update without the training marker; D stays in [0, 1] | §5.3, D2, D4, D15 |
| Slope | endpoints α_min and α_max; nondecreasing in D (strict for D > 0) for p ∈ {0.5, 1, 2, 4}; α_min = α_max gives LeakyReLU; bounded slope drift per step | §5.2, §6, P2, P5, P6 |
| Autograd | ∂out/∂z equals 1 or α_j; equals 1 at z = 0; never below α_min on z < 0; `gradcheck`; no parameters and no gradient into D; inactive units still pass gradient upstream | §6, P1 |
| Eval mode | state frozen even when marked; uses last training α; a sample's output does not depend on the rest of the batch | §5.3 |
| Checkpoints | D, update count and hyperparameters in `state_dict`; save/load round trip; mismatched hyperparameters rejected; stop-and-resume equals an uninterrupted run | Resume requirement |
| Dtype/device | `.half()` keeps D in float32; bfloat16 input works; CUDA equals CPU (Colab only) | D15 |
| Validation (new) | empty batches rejected before D is touched; NaN/Inf rejected in `d_init`, hyperparameters, `reset_state` and `load_state_dict`; `unit_dim` must be an int ≥ 1 and inside the input's rank; finite-value check on by default and switchable off without changing values | Review fixes |
| Optimizer integration (new) | SGD+momentum and Adam steps change weights; `num_updates` equals tracked steps; D equals a hand-recomputed EMA of the recorded preactivations; gradients reach all parameters and never D; eval and untracked passes leave D unchanged | §5.3, D2, D4 |
| Full resume (new) | model, optimizer, DARL state, step and torch/numpy/python RNG saved mid-run and loaded into fresh objects; losses, parameters, D, counters and optimizer state are bitwise equal to an uninterrupted run (SGD and Adam, two split points) | Resume requirement |

## How to use DARL in a training loop

```python
from darl_study.activations import DARL, darl_tracking

model = torch.nn.Sequential(torch.nn.Linear(784, 256), DARL(256), torch.nn.Linear(256, 10))

model.train()
with darl_tracking(model):          # marks this forward as a training step (updates D)
    loss = criterion(model(x), y)
loss.backward()
optimizer.step()

model.eval()                        # D frozen; metric passes never change it
```

### Finite-value checking

`DARL(..., check_finite=True)` is the default. Every forward pass raises `FloatingPointError` if the input contains NaN or Inf, before D is updated, so a diverging run stops at the first bad batch instead of silently writing a wrong state. Keep it on for debugging and pilot runs.

For timed performance experiments only, pass `check_finite=False`. The check reads the whole tensor and on CUDA forces a GPU-to-CPU sync each forward; switching it off never changes any computed value (a test checks this). Empty batches and bad `unit_dim` values are rejected either way.

A forward pass in train mode **without** `darl_tracking` leaves D unchanged by design (decision D4). The training pipeline in a later stage will always wrap its training forward in it.
