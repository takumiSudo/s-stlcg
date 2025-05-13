import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from formula import STLFormula, Always, Eventually, Until
import os

import matplotlib
matplotlib.use("Agg")                  # no display needed
import matplotlib.pyplot as plt

def plot_predicate(sp, fname: str, title: str = None):
    """
    Sample u ∈ [0,1], evaluate the aggregate spline predicate,
    and save to fname in the 'fig/' directory.
    """
    # Ensure the 'fig/' directory exists
    os.makedirs("fig", exist_ok=True)
    fname = os.path.join("fig", fname)

    # assume 1-D signal for visualization; we ignore other dims
    # build inputs of shape [N, signal_dim]
    N = 200
    u = torch.linspace(0.0, 1.0, N).unsqueeze(-1)  # [N, 1]
    # if signal_dim >1, pad zeros
    sig_dim = sp.predicate_net.proj.linear.in_features
    if sig_dim > 1:
        u = u.repeat(1, sig_dim)

    with torch.no_grad():
        y = sp.predicate_net(u)  # [N]

    plt.figure()
    plt.plot(u[:, 0].cpu().numpy(), y.cpu().numpy(), label="μ(u)")
    plt.xlabel("u")
    plt.ylabel("μ(u)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

class SmoothBSplineLayer(nn.Module):
    """
    Piecewise‐linear B‐spline layer with soft interpolation.
    - knots: Tensor [m_channels, K]
    - coefs: Tensor [m_channels, K]
    - tau:   smoothing temperature (>0)
    """
    def __init__(self, m_channels: int, n_knots: int, tau: float = 0.05):
        super().__init__()
        # initialize uniform knots/coefs as before
        initial_knots = torch.linspace(0.0, 1.0, n_knots)
        self.knots = nn.Parameter(initial_knots.unsqueeze(0)
                                  .repeat(m_channels, 1))    # [m, K]
        self.coefs = nn.Parameter(torch.zeros(m_channels, n_knots))
        self.tau   = tau

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [..., m_channels]
        # compute distance to each knot
        # [..., m_channels, K]
        u = u.contiguous()
        diff    = u.unsqueeze(-1) - self.knots  
        # soft‐assignment to each knot via L1‐distance kernel
        weights = torch.softmax(-diff.abs() / self.tau, dim=-1)
        # weighted sum of coefficients
        # [..., m_channels]
        return (weights * self.coefs).sum(dim=-1)

    def curvature_regularizer(self) -> torch.Tensor:
        """
        L2 penalty on second differences of coefficients:
          sum_q sum_i (c_q[i+2] - 2 c_q[i+1] + c_q[i])^2
        """
        c = self.coefs  # [m, K]
        # Δ² c[:, i] = c[:, i+2] - 2 c[:, i+1] + c[:, i]
        d2 = c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]
        return (d2**2).mean()

    def knot_spacing_regularizer(self) -> torch.Tensor:
        """
        Penalize uneven spacing between knots.
        """
        diffs     = self.knots[:, 1:] - self.knots[:, :-1]        # [m, K-1]
        mean_diff = diffs.mean(dim=1, keepdim=True)               # [m, 1]
        var       = ((diffs - mean_diff)**2).mean()
        return var


class KanProjection(nn.Module):
    """Inner linear projections: ℝⁿ → ℝᵐ."""
    def __init__(self, in_dim: int, m_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, m_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class KanSplinePredicate(nn.Module):
    """
    Full KAN–Spline atomic predicate with smoothing + reg helpers.
      μ(x) = ∑₍q₌0₎ᵐ⁻¹ Φ_q(λ_qᵀ x + b_q)
    """
    def __init__(self,
                 in_dim: int,
                 m_channels: int,
                 n_knots: int,
                 tau: float = 0.05):
        super().__init__()
        self.proj   = KanProjection(in_dim, m_channels)
        self.spline = SmoothBSplineLayer(m_channels, n_knots, tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.proj(x)             # [..., m_channels]
        v = self.spline(u)           # [..., m_channels]
        return v.sum(dim=-1)         # [...], scalar per time step

    def regularization(self,
                       λ_curv: float = 1e-2,
                       λ_knot: float = 1e-3) -> torch.Tensor:
        """
        Returns:
          λ_curv * curvature_reg + λ_knot * knot_spacing_reg
        """
        reg  = λ_curv * self.spline.curvature_regularizer()
        reg += λ_knot * self.spline.knot_spacing_regularizer()
        return reg


class SplinePredicate(STLFormula):
    """
    A leaf‐predicate that uses our learnable, smoothed KAN–Spline.
    """
    def __init__(self,
                 signal_dim: int,
                 m_channels: int   = None,
                 n_knots: int      = None,
                 tau: float        = 0.05):
        super().__init__()
        m_channels = m_channels or (2 * signal_dim + 1)
        n_knots    = n_knots    or 8

        self.predicate_net = KanSplinePredicate(
            in_dim     = signal_dim,
            m_channels = m_channels,
            n_knots    = n_knots,
            tau        = tau
        )

    def robustness_trace(self,
                         signal: torch.Tensor,
                         **kwargs) -> torch.Tensor:
        return self.predicate_net(signal)

    def _next_function(self):
        return self.predicate_net

    def predicate_regularization(self,
                                 λ_curv: float = 1e-2,
                                 λ_knot: float = 1e-3) -> torch.Tensor:
        """
        Expose a friendly method to get the reg penalty from your training loop.
        """
        return self.predicate_net.regularization(λ_curv, λ_knot)


# ─── Simple Pipeline Smoke‐Test ─────────────────────────────────────────────────

def _test_spline_pipeline():
    import torch
    import os

    # 1) Build a SplinePredicate for a 2‐dimensional signal
    sp = SplinePredicate(signal_dim=2,
                         m_channels=5,  # overparameterized, but fine
                         n_knots=8)

    # 2) Create a dummy signal: T time‐steps of 2D observations
    T = 10
    signal = torch.randn(T, 2, requires_grad=True)

    # 3) Forward: get a robustness trace of length T
    trace = sp.robustness_trace(signal)     # should be shape [T]
    assert trace.shape == (T,), f"Bad trace shape: {trace.shape}"

    # 4) Backward: check gradients flow back to the input
    loss = trace.mean()
    loss.backward()
    assert signal.grad is not None, "No gradient to input signal!"
    assert signal.grad.shape == (T, 2), f"Bad grad shape: {signal.grad.shape}"

    print("✅ SplinePredicate pipeline OK (forward + backward).")

# ─── Temporal Operators Smoke-Test ─────────────────────────────────────────────

def _test_spline_pipeline_temporal():

    # 1) create a dummy 2D signal of length T
    T = 10
    signal = torch.randn(T, 2, requires_grad=True)

    # 2) build two 1-D spline predicates (one per dimension)
    sp1 = SplinePredicate(signal_dim=2, m_channels=1, n_knots=5)
    sp2 = SplinePredicate(signal_dim=2, m_channels=1, n_knots=5)

    # 3) identity‐initialize them to x→x[ :,0] and x→x[:,1]
    with torch.no_grad():
        # proj: channel copies the respective input dim
        sp1.predicate_net.proj.linear.weight.zero_()
        sp1.predicate_net.proj.linear.weight[0, 0] = 1.0
        sp1.predicate_net.proj.linear.bias.zero_()
        sp2.predicate_net.proj.linear.weight.zero_()
        sp2.predicate_net.proj.linear.weight[0, 1] = 1.0
        sp2.predicate_net.proj.linear.bias.zero_()
        # spline: make Φ(u)=u by setting coefs=knots
        sp1.predicate_net.spline.coefs.copy_(sp1.predicate_net.spline.knots)
        sp2.predicate_net.spline.coefs.copy_(sp2.predicate_net.spline.knots)

    # Test Always
    phi_a = Always(sp1, interval=None)
    trace_a = phi_a.robustness_trace(signal)
    assert trace_a.shape == (T,), f"Always trace wrong shape: {trace_a.shape}"
    (trace_a.mean()).backward(retain_graph=True)
    assert signal.grad is not None, "No grad through Always"
    signal.grad.zero_()

    # Test Eventually
    phi_e = Eventually(sp1, interval=None)
    trace_e = phi_e.robustness_trace(signal)
    assert trace_e.shape == (T,), f"Eventually trace wrong shape: {trace_e.shape}"
    (trace_e.mean()).backward(retain_graph=True)
    assert signal.grad is not None, "No grad through Eventually"
    signal.grad.zero_()

    # Test Until over window [0,3]
    phi_u = Until(sp1, sp2, interval=[0, 3])
    trace_u = phi_u.robustness_trace(signal)
    assert trace_u.shape == (T,), f"Until trace wrong shape: {trace_u.shape}"
    (trace_u.mean()).backward()
    assert signal.grad is not None, "No grad through Until"

    print("✅ Temporal operators (Always, Eventually, Until) smoke-test passed")



# --- 1) a tiny “model” wrapping one SplinePredicate leaf ---
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 2D signals → one spline predicate → scalar per time step
        self.pred = SplinePredicate(signal_dim=2,
                                    m_channels=5,
                                    n_knots=8,
                                    tau=0.05)

    def forward(self, x):
        # x: [batch, T, 2]  →  trace: [batch, T]
        return self.pred.robustness_trace(x)

    def get_reg(self, λ_curv=1e-2, λ_knot=1e-3):
        # pull out the spline regularizer
        return self.pred.predicate_regularization(λ_curv, λ_knot)


# --- 2) dummy loss: MSE to random “labels” ---
def some_task_loss(trace, labels):
    # both [batch, T]
    return torch.nn.functional.mse_loss(trace, labels)


# --- 3) synthetic dataset ---
def make_dataloader(num_samples=100, T=10):
    # random signals & random targets
    signals = torch.randn(num_samples, T, 2)
    targets = torch.randn(num_samples, T)
    ds = TensorDataset(signals, targets)
    return DataLoader(ds, batch_size=10, shuffle=True)


# --- 4) the smoke‐test itself ---
def _test_training_loop():
    model      = DummyModel()
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = make_dataloader()

    model.train()
    plot_predicate(
        model.pred,
        fname="predicate_before_training.png",
        title="Predicate before training"
    )
    for epoch in range(10):
        epoch_loss = 0.0
        for batch_idx, (signal, labels) in enumerate(dataloader):
            # 1) forward + task loss
            trace  = model(signal)                    # [batch, T]
            first_trace = trace[0].detach().cpu().numpy()
            print(f"[Epoch {epoch} Batch {batch_idx}] robustness trace (sample 0):\n  {first_trace}")
            loss_t = some_task_loss(trace, labels)

            # 2) add spline‐regularization
            reg = model.get_reg(λ_curv=1e-2, λ_knot=1e-3)

            loss = loss_t + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch:>2} — avg loss: {epoch_loss / len(dataloader):.6f}")

    print("✅  Training‐loop smoke‐test passed")
    plot_predicate(
        model.pred,
        fname="predicate_after_training.png",
        title="Predicate after training"
    )


if __name__ == "__main__":
    _test_spline_pipeline()
    _test_spline_pipeline_temporal()
    _test_training_loop()