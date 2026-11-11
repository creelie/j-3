"""Illustration only (not used in any proof): one realisation of the
Carroll-Diachenko-Dulani simulation set-up, D=111, macrostates of dimension
1, 10, 100 in Haar-random position, energies iid uniform integers in [1,1000]
(units of omega_*, period T = 2*pi). Plots the Born weight p_1(t) of the
minimal macrostate and its cumulative time integral."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20261111)
D = 111
Z = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2)
Q, R = np.linalg.qr(Z)
U = Q * (np.diag(R) / np.abs(np.diag(R)))       # Haar unitary
E = rng.integers(1, 1001, size=D).astype(float)  # energy basis = standard basis
psi0 = U[:, 0]
q = np.abs(psi0) ** 2
T = 2 * np.pi
t = np.linspace(0, T, 400001)
amp = (q[None, :] * np.exp(-1j * np.outer(t, E))).sum(axis=1)
p1 = np.abs(amp) ** 2                          # d_1 = 1: p_1 = |<psi0|psi(t)>|^2
pbar_exact = sum(q[E == e].sum() ** 2 for e in np.unique(E))
dE = np.sqrt((q * E**2).sum() - (q * E).sum() ** 2)
win = (np.minimum(t, T - t) <= 3 / dE)
share = np.trapezoid(p1 * win, t) / np.trapezoid(p1, t)
cum = np.concatenate([[0], np.cumsum((p1[1:] + p1[:-1]) / 2 * np.diff(t))]) / (T * pbar_exact)
print(f"pbar={pbar_exact:.5f} mean={p1.mean():.5f} dE={dE:.1f} share(|t|<3/dE)={share:.3f}")

plt.rcParams.update({"font.size": 10, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(t / T, p1, lw=0.4, color="#1f4e79")
ax[0].axhline(pbar_exact, color="#c0392b", lw=1.0, ls="--", label=r"time average $\bar p_1$")
ax[0].axhline(1 / D, color="#7f8c8d", lw=1.0, ls=":", label=r"$d_1/D$")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 1.5)
ax[0].set_xlabel(r"$t/T_*$"); ax[0].set_ylabel(r"$p_1(t)$")
ax[0].set_title("Born weight of the minimal macrostate")
ax[0].legend(loc="lower center", fontsize=8, frameon=False)
ax[1].plot(t / T, cum, color="#1f4e79", lw=1.2)
ax[1].plot([0, 1], [0, 1], color="#7f8c8d", lw=0.8, ls=":")
ax[1].set_xlabel(r"$t/T_*$"); ax[1].set_ylabel(r"$\int_0^t p_1\,/\,(T_*\bar p_1)$")
ax[1].set_title("Cumulative share of that weight")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_weight.png", dpi=220)
