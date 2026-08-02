# Mathematical method and algorithms

## 1. Parameterization

For separation $s$, mass ratio $q$, and normalized source radius $\rho$, let

$$
A_{s,q,\rho}(x,y)
$$

be the finite-source binary-lens magnification returned by VBMicrolensing. In
polar coordinates around the map-frame origin, define the excess
magnification

$$
X(r,\phi)=A(r\cos\phi,r\sin\phi)-1.
$$

Hammlet stores angular Fourier coefficients at fixed radial nodes $r_j$:

$$
c_m(r_j)=\frac{1}{2\pi}\int_0^{2\pi}
X(r_j,\phi)e^{-im\phi}\,d\phi,
\qquad m=0,\ldots,M.
$$

Because $X$ is real,

$$
X_M(r,\phi)=c_0(r)+2\mathrm{Re}
\sum_{m=1}^{M} c_m(r)e^{im\phi}.
$$

Only non-negative modes are stored. Rotation of the source trajectory by
$\alpha$ is now diagonal in mode space:

$$
c_m(r)\longmapsto c_m(r)e^{im\alpha}.
$$

This identity is the central reason all trial angles can be evaluated together
by an inverse FFT.

## 2. Direct adaptive coefficient construction

No intermediate Cartesian magnification map is required. For each ring,
Hammlet starts from a uniform power-of-two angular grid. It computes the VBM
values, applies an `rfft`, doubles the grid, and compares the common retained
modes. A ring is globally converged when the series norm of the coefficient
change satisfies the configured absolute and relative tolerances.

Caustic polylines are transformed to the same map frame. Ring/caustic
intersections and near-tangent directions are detected. Guarded rings start at
a denser uniform grid and receive geometrically nested nonuniform samples on
both sides of each dangerous angle. For nonuniform nodes, the code integrates
the periodic piecewise-linear interpolant analytically rather than applying a
trapezoid rule. On an interval $[a,a+h]$, with endpoint values $v_0,v_1$, its
contribution to mode $m>0$ is

$$
\frac{e^{-ima}}{2\pi}\left[
v_0J_0(m,h)+\frac{v_1-v_0}{h}J_1(m,h)
\right],
$$

where

$$
J_0=\frac{1-e^{-imh}}{im},\qquad
J_1=-\frac{he^{-imh}}{im}+\frac{J_0}{im}.
$$

Thus local refinement does not leak modes merely because coarse and fine
intervals coexist.

## 3. Why outer polar rings are not undersampled

A fixed number of angles gives arc length $r\,\Delta\phi$, which grows with
radius. Hammlet addresses this in two complementary places:

1. Each `(s,q)` bucket evaluates sharp-rho pilot maps on a dense quadratic
   radial pilot grid. Leave-one-radius-out spectral interpolation error defines
   a difficulty density. Sixty-five percent of the fixed node budget follows
   its cumulative distribution; the remainder stays uniformly distributed so
   no outer interval can become arbitrarily wide.
2. Angular sampling is driven by retained-coefficient convergence and caustic
   proximity, not by a single fixed $N_\phi$. A caustic-guarded outer ring can
   therefore use the same fine angular work limit as an inner ring.
3. Every final radial interval receives nested direct-evaluator holdout rings.
   Their spectra define a denser piecewise-linear radial reference. A
   polynomial remainder bound certifies both the linear and cubic runtime
   interpolants against that reference and folds the result into the stored
   node-error envelope.

The per-ring generation cost remains bounded by `max_n_phi`. Radial holdouts
increase one-time generation work but add no search-time contraction. This is
an accuracy/work policy, not a proof that every continuous radial feature
between unevaluated holdout rings is resolved. See
[accuracy.md](accuracy.md).

## 4. Trajectory contraction

For a rectilinear seed $(t_0,u_0,t_E)$, define

$$
\tau_i=\frac{t_i-t_0}{t_E},\qquad
r_i=\sqrt{\tau_i^2+u_0^2},\qquad
\psi_i=\mathrm{atan2}(-u_0,-\tau_i).
$$

The trajectory angle $\alpha$ shifts $\psi_i$ by $\alpha$. Radial interpolation is
a fixed linear stencil:

$$
\widetilde c_{im}=\sum_j L_{ij}c_m(r_j).
$$

Consequently

$$
x_i(\alpha)=\widetilde c_{i0}+
2\mathrm{Re}\sum_{m=1}^{M}
\widetilde c_{im}e^{im\psi_i}e^{im\alpha}.
$$

Weighted sums over observations can be performed before the $\alpha$ inverse FFT.
This changes the expensive dimension from
$(\mathrm{map},\alpha,\mathrm{observation})$ to compact Fourier moments over
$(\mathrm{map},\mathrm{radial\ node},\mathrm{mode})$. The FFT itself is only part of the speedup; the
larger gain is that the precontracted event kernel removes repeated adaptive
map traversal, interpolation, and observation loops.

## 5. Analytic flux profiling and chi-square

For dataset $d$, the model is

$$
F_i=F_{s,d}A_i+F_{b,d}.
$$

At every $(\mathrm{map},\mathrm{geometry},\alpha)$, Hammlet profiles the two linear fluxes
exactly. Let $w_i=\sigma_i^{-2}$ and

$$
G=\begin{pmatrix}
\sum w_i A_i^2 & \sum w_iA_i\\
\sum w_iA_i & \sum w_i
\end{pmatrix},\qquad
b=\begin{pmatrix}
\sum w_iA_iF_i\\
\sum w_iF_i
\end{pmatrix}.
$$

Then

$$
\hat\beta=(\hat F_s,\hat F_b)^T=G^{-1}b,
\qquad
\chi^2_{\min}=\sum_iw_iF_i^2-b^TG^{-1}b.
$$

The required linear and quadratic magnification moments are Fourier series.
Products use coefficient convolution; the implementation stores/constructs the
necessary lag terms so the same radially interpolated light curve supplies both
$A$ and $A^2$. This avoids an inconsistent approximation in the normal matrix.

Multiple datasets add their independently profiled chi-square values, while a
geometry batch shares the same map coefficients.

## 6. Multi-resolution selection

The base pass uses $(M_b,N_{\alpha,b})$, scans every map, and records the central
chi-square plus an uncertainty interval. The full pass uses
$\left(M_f,N_{\alpha,f}\right)$ only on a union of:

- the best base chi-square maps;
- the largest spectral-risk maps;
- every interval that can overlap the current top-K upper cutoff when
  `certified_selection=True`.

If $U_{(K)}$ is the $K$-th smallest upper bound, interval-safe selection retains
every candidate $i$ for which

$$
L_i\le U_{(K)}.
$$

This prevents a hard central-chi-square cutoff from discarding a candidate that
can still enter the true top K under the supplied error envelope.

## 7. High-mode seed refinement

After interval-safe selection and clustering, the normal handoff objective uses
all stored modes (normally $M=512$) and cubic radial interpolation. For parameter vector

$$
p=(t_0,u_0,\log t_E,\alpha),
$$

a JAX-batched coordinate pattern search tests the incumbent and both directions
of every axis. Two shallow levels cost 17 evaluations per seed. The best 32
seeds also race their discrete $(s,q,\rho)$ neighbours, sweep six correlated
axis pairs, and run ten deeper levels. This remains a refinement of the stored
Fourier maps, not a continuous binary-lens physical fit.

## 8. Complexity

For $N_{\rm map}$ maps, $N_r$ radial nodes, $M$ modes, $N_{\rm obs}$ observations and
$N_{\alpha}$ angles:

- map construction is dominated by direct VBM calls and occurs once;
- event-kernel construction is approximately $O(N_{\rm obs}M)$ per geometry;
- map contraction is approximately $O(N_{\rm map}N_rM)$;
- the angular transform is approximately $O(N_{\rm map}N_{\alpha}\log N_{\alpha})$.

A naive direct loop instead repeats VBM/map interpolation for roughly
$N_{\rm map}N_{\alpha}N_{\rm obs}$ points. Map-coefficient arrays are memory-mapped and
sharded, allowing bounded scans and sequential storage access.
