import torch
import torch.nn as nn
import torch.nn.functional as F
from formula import STLFormula

def b_spline_eval(u: torch.Tensor,
                  knots: torch.Tensor,
                  coefs: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Piecewise‐linear spline evaluation via linear interpolation.
    Args:
      u:        Tensor of shape [..., m_channels], points to evaluate.
      knots:    Tensor of shape [m_channels, K], assumed sorted ascending.
      coefs:    Tensor of shape [m_channels, K], values at each knot.
      eps:      Small constant to avoid division by zero.
    Returns:
      Tensor of shape [..., m_channels]: interpolated values.
    """
    orig_shape = u.shape
    m_channels, K = knots.shape
    # Flatten leading dims
    flat_u = u.reshape(-1, m_channels)  # [N, m_channels]
    outputs = torch.empty_like(flat_u)

    for q in range(m_channels):
        u_q = flat_u[:, q]                        # [N]
        k_q = knots[q]                            # [K]
        c_q = coefs[q]                            # [K]
        # Find right‐bin indices: idx in [0..K]; idx=0 means u < k_q[0], idx=K means u>=k_q[-1]
        idx = torch.bucketize(u_q, k_q)           # [N], values in 0..K
        # Clip so that idx0 in [0..K-2], idx1=idx0+1
        idx0 = (idx - 1).clamp(0, K-2)
        idx1 = (idx0 + 1)
        k0 = k_q[idx0]                            # [N]
        k1 = k_q[idx1]                            # [N]
        c0 = c_q[idx0]                            # [N]
        c1 = c_q[idx1]                            # [N]
        # Compute linear interp weight
        t = (u_q - k0) / (k1 - k0 + eps)          # [N]
        outputs[:, q] = c0 + t * (c1 - c0)

    return outputs.reshape(orig_shape)            # [..., m_channels]

class KanProjection(nn.Module):
    """Inner linear projections: ℝⁿ → ℝᵐ (m ≥ 2n+1)."""
    def __init__(self, in_dim: int, m_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, m_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim] → returns [..., m_channels]
        return self.linear(x)

class BSplineLayer(nn.Module):
    """Piecewise‐linear B‐spline layer on each channel in parallel."""
    def __init__(self, m_channels: int, n_knots: int):
        super().__init__()
        # Initialize uniform knot positions in [0,1]
        initial_knots = torch.linspace(0.0, 1.0, n_knots)
        self.knots = nn.Parameter(initial_knots.unsqueeze(0)
                                  .repeat(m_channels, 1))    # [m_channels, K]
        # Initialize coefficients to zero (predicts zero until trained)
        self.coefs = nn.Parameter(torch.zeros(m_channels, n_knots))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [..., m_channels]
        return b_spline_eval(u, self.knots, self.coefs)     # [..., m_channels]

class KanSplinePredicate(nn.Module):
    """
    Full KAN–Spline atomic predicate:
      μ(x) = ∑₍q₌0₎ᵐ⁻¹ Φ_q(λ_qᵀ x + b_q),
    where Φ_q are learned piecewise‐linear splines.
    """
    def __init__(self,
                 in_dim: int,
                 m_channels: int,
                 n_knots: int):
        """
        Args:
          in_dim:      dimension of the input signal vector x(t).
          m_channels:  number of spline channels (≥ 2·in_dim+1).
          n_knots:     number of knots per spline.
        """
        super().__init__()
        self.proj   = KanProjection(in_dim, m_channels)
        self.spline = BSplineLayer(m_channels, n_knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [..., in_dim] input signal at a single time-step.
        Returns:
          [...]: scalar robustness score per time-step.
        """
        u = self.proj(x)             # [..., m_channels]
        v = self.spline(u)           # [..., m_channels]
        return v.sum(dim=-1)         # [...], summed over channels


class SplinePredicate(STLFormula):
    """
    A leaf‐predicate that uses a learnable KAN‐Spline instead of a fixed linear test.
    """

    def __init__(self,
                 signal_dim: int,
                 m_channels: int   = None,
                 n_knots: int      = None):
        """
        Args:
          signal_dim:  dimensionality of each time‐slice x(t)
          m_channels:  # spline channels (default: 2*signal_dim+1)
          n_knots:     # knots per spline (default: 8)
        """
        super().__init__()
        # sensible defaults
        m_channels = m_channels or (2 * signal_dim + 1)
        n_knots    = n_knots    or 8

        # replace old a, b parameters with our Kan–Spline net
        self.predicate_net = KanSplinePredicate(
            in_dim     = signal_dim,
            m_channels = m_channels,
            n_knots    = n_knots
        )

    def robustness_trace(self,
                         signal: torch.Tensor,
                         **kwargs) -> torch.Tensor:
        """
        signal: [T, signal_dim]
        returns: [T] robustness at each time-step
        """
        # vectorized: KanSplinePredicate accepts [..., signal_dim]
        # so we can just do:
        #   outputs = self.predicate_net(signal)
        # which yields shape [T]
        return self.predicate_net(signal)

    def _next_function(self):
        # if you need to visualize subformulas, keep this
        return self.predicate_net

# ─── Simple Pipeline Smoke‐Test ─────────────────────────────────────────────────

def _test_spline_pipeline():
    import torch

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
    import torch
    from spline import SplinePredicate
    from formula import Always, Eventually, Until

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

if __name__ == "__main__":
    _test_spline_pipeline()
    _test_spline_pipeline_temporal()