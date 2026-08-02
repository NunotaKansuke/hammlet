# Accuracy certificates and limitations

## Angular certificate

Let $f(\phi)$ be the periodic piecewise-linear interpolant through the final
adaptive VBM samples, and let $P_D(\phi)$ be the diagnostic Fourier polynomial
retained through mode $D$. On one sampling interval of width $h$, $f''=0$, so
the residual $r=f-P_D$ obeys $r''=-P_D''$. Linear interpolation's standard
remainder inequality gives

$$
\max_I |r|\le
\max(|r(a)|,|r(a+h)|)+\frac{h^2}{8}\|P_D''\|_{\infty,I}.
$$

The global derivative norm has the computable upper bound

$$
\|P_D''\|_\infty
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

## Radial certificate

Let $r_j$ be the stored radial nodes and let $\hat r_k$ contain both those nodes
and the configured dyadic direct-VBM holdout rings. At each $\hat r_k$, the
angular construction supplies coefficients $d_m(\hat r_k)$ and a certified
angular remainder $e_k$. Between adjacent reference rings, Hammlet defines
$d_m(r)$ by linear interpolation.

The runtime coefficient $p_m(r)$ is the fixed linear or cubic Lagrange
interpolant through the stored complex64 coefficients. On one reference segment
$I=[a,a+h]$, the residual $g_m=d_m-p_m$ obeys $g_m''=-p_m''$. Consequently,

$$
\max_I |g_m|
\le
\max\!\left(|g_m(a)|,|g_m(a+h)|\right)
+\frac{h^2}{8}\max_I|p_m''|.
$$

For cubic interpolation, $p_m''$ is linear, so its absolute maximum is bounded
by its two endpoint values. For linear interpolation the derivative term is
zero. Summing the coefficient bounds with Fourier weights gives

$$
R_I=B_{I,0}+2\sum_{m=1}^{M}B_{I,m}+\max(e_a,e_{a+h}).
$$

Every node read by the interval's runtime stencil receives at least $R_I$.
Because Lagrange weights sum to one, their absolute values sum to at least one;
the existing absolute-stencil propagation therefore encloses the complete
radial interval, including negative cubic weights. The stored node envelope is
the maximum needed by either the linear ranking pass or cubic full pass.

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

The current guarantee is conditional on a sampled two-dimensional reference:

- **Included:** every angle and radius of the declared angular/radial
  piecewise-linear VBM sample reference, retained diagnostic tail, nested
  coefficient change, radial Lagrange interpolation, and complex64 storage
  rounding.
- **Not included:** unknown VBM variation between adjacent radial holdout rings.
- **Not included:** a formal interval bound on VBMicrolensing's own internal
  finite-source numerical integration error.

Thus the interval is useful and mathematically constructed, but must not be
described as a proof against continuous direct VBM at every $(r,\phi)$. A future
continuous-VBM certificate still needs a valid radial derivative enclosure from
the underlying solver. Scientific finalists must be directly evaluated with
VBMicrolensing.

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
