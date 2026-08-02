# Accuracy certificates and limitations

## Angular certificate

Let $f(\phi)$ be the periodic piecewise-linear interpolant through the final
adaptive VBM samples, and let $P_D(\phi)$ be the diagnostic Fourier polynomial
retained through mode $D$. On one sampling interval of width $h$, $f''=0$, so
the residual $r=f-P_D$ obeys $r''=-P_D''$. Linear interpolation's standard
remainder inequality gives

$$
\max_I |r|\le
\max(|r(a)|,|r(a+h)|)+\frac{h^2}{8}\lVert P_D''\rVert_{\infty,I}.
$$

The global derivative norm has the computable upper bound

$$
\lVert P_D''\rVert_\infty
\le 2\sum_{m=1}^{D}m^2|c_m|.
$$

For a runtime mode budget $M \le D$, Hammlet adds the exactly known omitted
diagnostic tail,

$$
E_M = E_D + 2\sum_{m=M+1}^{D}|c_m|,
$$

plus the nested-grid coefficient-change envelope and outward-rounded complex64
storage error. Therefore $|f(\phi)-P_M(\phi)| \le E_M$ for every angle, relative to
the declared sampled reference.

## Propagation to chi-square

Radial interpolation propagates node errors with absolute stencil weights. The
event kernel precomputes the corresponding weighted absolute centered-flux and
quadratic lag moments. Triangle and Cauchy-type inequalities then bound the
perturbation of the profiled normal-equation moments, producing generally
asymmetric

$$
\chi^2_{\rm lower}\le\chi^2_{\rm reference}\le\chi^2_{\rm upper}.
$$

The bound calculation reuses compact event moments; it does not allocate a
map-by-observation error matrix. In the development benchmarks inherited by
this implementation, enabling the propagation added about 2--3% to the JAX
scan time.

## What is and is not guaranteed

The current guarantee is conditional and angular:

- **Included:** every angle of the periodic piecewise-linear VBM sample
  reference, retained diagnostic tail, nested coefficient change, and
  complex64 storage rounding.
- **Not yet included:** unknown VBM variation between adjacent radial nodes.
- **Not included:** a formal interval bound on VBMicrolensing's own internal
  finite-source numerical integration error.

Thus the interval is useful and mathematically constructed, but must not be
described as a proof against continuous direct VBM at every $(r,\phi)$. A future
two-dimensional certificate needs adaptive radial midpoint/holdout rings or a
valid radial derivative enclosure. Scientific finalists must be directly
evaluated with VBMicrolensing.

## Choosing M

Increasing $M$ raises contraction and inverse-FFT work but shrinks the known
tail and therefore the number of overlapping intervals that require a full
rescan. The optimal setting minimizes

$$
T(M)=T_{\rm base}(M)+N_{\rm rescue}(M)T_{\rm full/map}.
$$

Hammlet stores a larger production budget (`M=512`) while defaulting the search
to a cheap base pass (`M=32`), a targeted cubic pass (`M=128`), and high-mode
seed refinement using all stored modes (`M=512`). Users should
benchmark the frontier on representative smooth, central-caustic, and
planetary-caustic events before changing production defaults.
