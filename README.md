# DARL study — Stage 1: DARL module and unit tests

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
    └── test_darl.py                  # 55 unit tests (54 run on CPU, 1 needs CUDA)
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

```powershell
python -m pytest -v
python scripts\darl_demo.py
```

In PyCharm you can also right-click `tests/test_darl.py` → **Run 'pytest in test_darl.py'**.

## Expected output and pass criteria

**Tests:** the last line should read

```
54 passed, 1 skipped in X.XXs
```

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

A forward pass in train mode **without** `darl_tracking` leaves D unchanged by design (decision D4). The training pipeline in a later stage will always wrap its training forward in it.
