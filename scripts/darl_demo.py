"""Stage 1 sanity demo: how DARL's state and slope evolve for three units.

Unit 0 is negative on every input (a "dead" unit), unit 1 is positive on every
input, and unit 2 is negative on exactly half of each batch. Run with:

    python scripts/darl_demo.py
"""

import torch

from darl_study.activations import DARL, darl_tracking


def main() -> None:
    torch.manual_seed(0)
    m = DARL(3, alpha_min=0.01, alpha_max=0.3, p=1.0, rho=0.99)
    m.train()
    z = torch.tensor([[-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])  # r = [1.0, 0.0, 0.5]

    print("device: cpu | torch", torch.__version__)
    print(f"{'step':>5} | {'D_0':>7} {'D_1':>7} {'D_2':>7} | {'a_0':>7} {'a_1':>7} {'a_2':>7}")
    for step in range(1, 1001):
        with darl_tracking(m):
            m(z)
        if step in (1, 10, 100, 500, 1000):
            d, a = m.D.tolist(), m.alpha.tolist()
            print(f"{step:>5} | {d[0]:7.4f} {d[1]:7.4f} {d[2]:7.4f} | {a[0]:7.4f} {a[1]:7.4f} {a[2]:7.4f}")

    m.eval()
    before = m.D.clone()
    m(-torch.ones(4, 3))
    print("eval pass changed state:", not torch.equal(before, m.D))


if __name__ == "__main__":
    main()
