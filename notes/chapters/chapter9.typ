#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *
#import "@preview/algorithmic:1.0.2"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

= Local Polynomial Approximation
*Local Polynomial Approximation (LPA)* represents a fundamental shift in signal processing methodology, moving away from global sparsity constraints toward localized polynomial modeling. This approach recognizes that many real-world signals exhibit piecewise smooth behavior that can be effectively captured through local polynomial representations.

Traditional signal processing approaches often rely on global assumptions about signal structure, such as sparsity in transformed domains (e.g., DCT, wavelet transforms). While these methods have proven successful in many applications, they impose uniform constraints across the entire signal domain. In contrast, LPA methods adapt to local signal characteristics, providing more flexible and often more accurate approximations.

The transition from sparsity-based to polynomial-based modeling represents a significant conceptual advancement:
- *Sparsity-based approach*: Assumes signal can be represented as a linear combination of few basis functions from a fixed dictionary
- *Polynomial-based approach*: Assumes signal can be locally approximated by polynomials of appropriate degree

This shift enables adaptive processing where each spatial location can have its own optimal approximation parameters, leading to superior performance in regions with varying signal characteristics.

#definition("Local Polynomial Approximation")[
  Given a signal $f: RR -> RR$ and a point $x_0 in RR$, a local polynomial approximation of degree $L$ is a polynomial $P_L (x)$ of the form:
  $
    P_L (x) = sum_(j=0)^L beta_j (x - x_0)^j
  $
  that minimizes a local fitting criterion within a neighborhood of $x_0$.
] <lpa>

== Local Polynomial Model
Consider a one-dimensional signal model:
$
  y(t) = f(t) + eta(t)
$ <signal_model>
where $f(t)$ represents the underlying smooth signal and $eta(t)$ denotes additive white Gaussian noise with variance $sigma^2$.

For discrete processing, we work with sampled versions. Let $bold(y) = [y_1, y_2, ..., y_M]^T$ represent an $M$-dimensional signal vector extracted from a local neighborhood around a central pixel.

Within a local neighborhood, we assume the signal can be well approximated by a polynomial of degree $L$:
$
  f(t_i) approx sum_(j=0)^L beta_j t_i^j, quad i = 1, 2, ..., M
$ <poly_approximation>
where ${beta_j}_(j=0)^L$ are the polynomial coefficients to be estimated, and ${t_i}_(i=1)^M$ are the spatial locations within the neighborhood.

#attention("Degree Constraint")[
  The polynomial degree $L$ must satisfy $L + 1 <= M$ to ensure an over-determined system. This constraint prevents overfitting and ensures stable coefficient estimation.
]

=== Matrix Formulation
The polynomial approximation can be expressed in matrix form. Define the design matrix $bold(T) in RR^(M times (L+1))$:

$
  bold(T) = mat(
    1, t_1, t_1^2, dots.c, t_1^L;
    1, t_2, t_2^2, dots.c, t_2^L;
    dots.v, dots.v, dots.v, dots.down, dots.v;
    1, t_M, t_M^2, dots.c, t_M^L
  )
$ <design_matrix>

The polynomial coefficient vector is $bold(beta) = [beta_0, beta_1, ..., beta_L]^T in RR^(L+1)$.

The polynomial approximation becomes:
$
  bold(f) approx bold(T) bold(beta)
$ <matrix_poly>

=== Least Squares Formulation
The optimal polynomial coefficients are obtained by minimizing the squared approximation error:
$
  hat(bold(beta)) = argmin_(bold(beta) in RR^(L+1)) norm(bold(y) - bold(T) bold(beta))^2
$ <ls_objective>
#theorem("Unweighted LPA Solution")[
  The least squares solution to the local polynomial approximation problem is:
  $ hat(bold(beta)) = (bold(T)^T bold(T))^(-1) bold(T)^T bold(y) $ <ls_solution>
  provided that $bold(T)^T bold(T)$ is invertible.
] <unweighted_lpa>

*Proof:*
Taking the derivative of the objective function @ls_objective with respect to $bold(beta)$ and setting it to zero:
$
  (partial)/(partial bold(beta)) norm(bold(y) - bold(T) bold(beta))^2 &= (partial)/(partial bold(beta)) (bold(y) - bold(T) bold(beta))^T (bold(y) - bold(T) bold(beta)) \
  &= -2 bold(T)^T (bold(y) - bold(T) bold(beta)) = 0
$

Solving for $bold(beta)$ yields the normal equations:
$ bold(T)^T bold(T) bold(beta) = bold(T)^T bold(y) $

The solution follows directly when $bold(T)^T bold(T)$ is invertible.

#pagebreak()

== Weighted Local Polynomial Approximation
In many practical scenarios, not all samples within a neighborhood should contribute equally to the polynomial fit. Samples closer to the center of the neighborhood are typically more relevant for estimating the signal at that location. Weighted LPA addresses this by introducing spatially varying weights.

The weighted LPA problem incorporates a diagonal weight matrix $bold(W) in RR^(M times M)$:
$
  hat(bold(beta))_w = argmin_(bold(beta) in RR^(L+1)) norm(bold(W)(bold(y) - bold(T) bold(beta)))^2
$ <weighted_objective>
where $bold(W) = diag(w_1, w_2, ..., w_M)$ with $w_i >= 0$ for all $i$.

#theorem("Weighted LPA Solution")[
  The weighted least squares solution is:
  $ hat(bold(beta))_w = (bold(T)^T bold(W)^2 bold(T))^(-1) bold(T)^T bold(W)^2 bold(y) $ <weighted_solution>
] <weighted_lpa_thm>

*Proof:*
The weighted objective function can be written as:
$ norm(bold(W)(bold(y) - bold(T) bold(beta)))^2 = norm(bold(W) bold(y) - bold(W) bold(T) bold(beta))^2 $

This is equivalent to solving the unweighted problem:
$ argmin_(bold(beta)) norm(tilde(bold(y)) - tilde(bold(T)) bold(beta))^2 $

where $tilde(bold(y)) = bold(W) bold(y)$ and $tilde(bold(T)) = bold(W) bold(T)$.

Applying @unweighted_lpa:
$
  hat(bold(beta))_w & = (tilde(bold(T))^T tilde(bold(T)))^(-1) tilde(bold(T))^T tilde(bold(y))           \
                    & = ((bold(W) bold(T))^T (bold(W) bold(T)))^(-1) (bold(W) bold(T))^T bold(W) bold(y) \
                    & = (bold(T)^T bold(W)^2 bold(T))^(-1) bold(T)^T bold(W)^2 bold(y)
$
Common weight functions include:
+ *Uniform weights*: $w_i = 1$ for all $i$ (reduces to unweighted case)
+ *Gaussian weights*: $w_i = exp(-(t_i - t_0)^2 slash (2 sigma_w^2))$
+ *Binary weights*: $w_i in {0, 1}$ for adaptive support selection

#example("Binary Weight Example")[
  Consider a signal with a discontinuity. Using binary weights allows selective processing:
  - Left-side filter: $bold(w) = [1, 1, 1, 0, 0]^T$
  - Right-side filter: $bold(w) = [0, 0, 1, 1, 1]^T$

  This prevents blurring across discontinuities while maintaining smoothing within homogeneous regions.
] <binary_weights>

#pagebreak()

== QR Decomposition Approach
Direct computation of $(bold(T)^T bold(T))^(-1)$ can be numerically unstable, especially when $bold(T)$ is ill-conditioned. The QR decomposition provides a numerically stable alternative while revealing the underlying geometric structure of the problem.

#theorem("QR Decomposition")[
  Any matrix $bold(T) in RR^(M times (L+1))$ with $M >= L+1$ and full column rank can be decomposed as:
  $ bold(T) = bold(Q) bold(R) $ <qr_decomp>
  where $bold(Q) in RR^(M times (L+1))$ has orthonormal columns ($bold(Q)^T bold(Q) = bold(I)_(L+1)$) and $bold(R) in RR^((L+1) times (L+1))$ is upper triangular.
] <qr_decomp_thm>

Using the QR decomposition, the LPA solution becomes:
$
  hat(bold(beta)) & = (bold(T)^T bold(T))^(-1) bold(T)^T bold(y)                               \
                  & = ((bold(Q) bold(R))^T (bold(Q) bold(R)))^(-1) (bold(Q) bold(R))^T bold(y) \
                  & = (bold(R)^T bold(Q)^T bold(Q) bold(R))^(-1) bold(R)^T bold(Q)^T bold(y)   \
                  & = (bold(R)^T bold(R))^(-1) bold(R)^T bold(Q)^T bold(y)                     \
                  & = bold(R)^(-1) bold(Q)^T bold(y)
$

The signal estimate is:
$
  hat(bold(f)) = bold(T) hat(bold(beta)) = bold(Q) bold(R) bold(R)^(-1) bold(Q)^T bold(y) = bold(Q) bold(Q)^T bold(y)
$ <signal_estimate_qr>

#proposition("Projection Interpretation")[
  The matrix $bold(P) = bold(Q) bold(Q)^T$ represents the orthogonal projection onto the column space of $bold(T)$. The LPA estimate is the orthogonal projection of the noisy signal onto the space of polynomials of degree $L$.
] <projection>

For weighted LPA, we apply QR decomposition to the weighted design matrix $bold(W) bold(T)$:
$
  bold(W) bold(T) = tilde(bold(Q)) tilde(bold(R))
$ <weighted_qr_eq>

The weighted solution becomes:
$
  hat(bold(beta))_w & = tilde(bold(R))^(-1) tilde(bold(Q))^T bold(W) bold(y) \
     hat(bold(f))_w & = tilde(bold(Q)) tilde(bold(Q))^T bold(W) bold(y)
$

#pagebreak()

== Convolution Implementation
A key insight in LPA is that the estimation can be implemented as a convolution operation, enabling efficient computation across entire signals or images.

Consider the estimation of the signal value at the center of the neighborhood. Let $i_c$ denote the central index. The estimate at this location is:
$
  hat(f)(t_(i_c)) = bold(e)_(i_c)^T bold(Q) bold(Q)^T bold(y)
$ <central_estimate>
where $bold(e)_(i_c)$ is the unit vector with 1 in the $i_c$-th position.

#theorem("Convolution Kernel Formula")[
  The LPA estimation at the central location can be expressed as:
  $ hat(f)(t_(i_c)) = sum_(i=1)^M h_i y_i = bold(h)^T bold(y) $ <conv_formula>
  where the convolution kernel is:
  $ bold(h) = bold(Q) bold(Q)^T bold(e)_(i_c) $ <kernel_formula>
] <conv_kernel>

The convolution kernel can be computed explicitly as:
$
  bold(h) = sum_(j=0)^L beta_j bold(q)_j
$ <explicit_kernel_eq>
where $bold(q)_j$ are the columns of $bold(Q)$ and:
$
  beta_j = bold(q)_j^T bold(e)_(i_c) = q_(j,i_c)
$ <kernel_coefficients>

== Special Cases <special_cases>

#example("Zero-Order Polynomial (Moving Average)")[
  For $L = 0$, the design matrix is $bold(T) = bold(1) = [1, 1, ..., 1]^T$.

  The QR decomposition gives:
  $
    bold(Q) & = 1/sqrt(M) bold(1) \
    bold(R) & = sqrt(M)
  $

  The convolution kernel becomes:
  $ bold(h) = 1/M bold(1) $

  This is the standard moving average filter.
] <zero_order>

#example("Weighted Zero-Order Polynomial")[
  For weighted zero-order polynomial with normalized weights $sum_(i=1)^M w_i^2 = 1$:

  $
    tilde(bold(Q)) & = bold(w) \
    tilde(bold(R)) & = 1
  $

  The convolution kernel is:
  $ bold(h) = bold(w) $

  This shows that Gaussian smoothing corresponds to weighted zero-order polynomial fitting.
] <weighted_zero_order>

== Statistical Properties and Performance Analysis
=== Bias-Variance Decomposition <bias_variance>
The mean squared error (MSE) of the LPA estimator can be decomposed into bias and variance components:
$
  MSE(hat(f)(t_0)) = Bias^2(hat(f)(t_0)) + Var(hat(f)(t_0))
$ <mse_decomposition>

=== Bias Analysis <bias_analysis>
#theorem("Bias of LPA Estimator")[
  For a signal $f(t)$ that is $(L+1)$-times differentiable, the bias of the LPA estimator is:
  $ Bias(hat(f)(t_0)) = (f^((L+1))(t_0))/((L+1)!) sum_(i=1)^M h_i (t_i - t_0)^(L+1) + O(h^(L+2)) $ <bias_formula>
  where $h$ is the neighborhood size.
] <bias_lpa>

#theorem("Bias for Polynomial Signals")[
  If the true signal is a polynomial of degree $L$ or less, the LPA estimator is unbiased.
] <poly_bias>

=== Variance Analysis <variance_analysis>
#theorem("Variance of LPA Estimator")[
  For additive white Gaussian noise with variance $sigma^2$, the variance of the LPA estimator is:
  $
    Var(hat(f)(t_0)) = sigma^2 norm(bold(h))^2 = sigma^2 bold(e)_(i_c)^T bold(Q) bold(Q)^T bold(e)_(i_c)
  $ <variance_formula>
] <variance_lpa>

The variance decreases as the effective number of samples increases, but the bias may increase due to the larger neighborhood size. This creates a fundamental bias-variance tradeoff.

=== Optimal Neighborhood Selection <optimal_neighborhood>
The optimal neighborhood size balances bias and variance:
$
  h_"opt" = argmin_h [Bias^2(h) + Var(h)]
$ <optimal_tradeoff>
In practice, this leads to adaptive algorithms that select different neighborhood sizes and shapes based on local signal characteristics.

#pagebreak()

== Adaptive Neighborhood Selection
Fixed neighborhood sizes and shapes are suboptimal for signals with varying local characteristics. Adaptive methods adjust the approximation parameters based on local signal properties.

=== Directional Filtering <directional>
Binary weights enable directional filtering, which is particularly useful near discontinuities:

#definition("Directional Kernels")[
  A set of directional kernels ${bold(h)_d}_(d=1)^D$ provides estimates along different directions or orientations. Each kernel uses binary weights to select samples from a specific spatial direction.
] <directional_kernels>

#example("One-Dimensional Directional Kernels")[
  For a 1D signal with neighborhood size $M = 5$:
  - $bold(h)_"left"$: weights $[1, 1, 1, 0, 0]$
  - $bold(h)_"right"$: weights $[0, 0, 1, 1, 1]$
  - $bold(h)_"center"$: weights $[0, 1, 1, 1, 0]$
] <1d_directional>

=== Intersection of Confidence Intervals
The Intersection of Confidence Intervals (ICI) rule provides a principled approach for adaptive neighborhood selection:

1. Compute estimates ${hat(f)_d}$ and confidence intervals ${"CI"_d}$ for each directional kernel
2. Find the intersection of all confidence intervals: $"CI"_"intersect" = inter.big_(d=1)^D "CI"_d$
3. Select the estimator with the largest neighborhood whose confidence interval contains $"CI"_"intersect"$

== The Intersection of Confidence Intervals (ICI) Rule
The ICI rule provides a data-driven approach to support selection based on statistical confidence intervals:

#definition("Confidence Interval for LPA")[
  For a given support size $h$ and confidence parameter $gamma > 0$, the confidence interval for $hat(f)_h (x_0)$ is:
  $
    "CI"_h (x_0) = [hat(f)_h (x_0) - gamma sqrt(V(h, x_0)), hat(f)_h (x_0) + gamma sqrt(V(h, x_0))]
  $
]

For Gaussian noise with known variance $sigma^2$, choosing $gamma = z_(alpha slash 2)$ (the $alpha slash 2$ quantile of the standard normal distribution) yields a $(1-alpha)$ confidence interval.

#theorem("ICI Principle")[
  Consider a sequence of support sizes $h_1 < h_2 < dots.c < h_K$. Define:
  $
    cal(I)_k (x_0) = inter.big_(j=1)^k "CI"_(h_j) (x_0)
  $
  The optimal scale $k^* (x_0)$ is chosen as:
  $
    k^* (x_0) = max{k : cal(I)_k (x_0) != emptyset}
  $
]

#pagebreak()

#algorithm-figure("Intersection of Confidence Intervals (ICI)", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure(
    [ICI],
    ([signal $bold(y) in RR^n$], [polynomial degree $p$], [scales ${h_k}_(k=1)^K$], [parameter $gamma$]),
    LineBreak,
    {
      Comment[*Input:* Signal $bold(y) in RR^n$, polynomial degree $p$, scales ${h_k}_(k=1)^K$, parameter $gamma$]
      LineBreak
      Comment[*Output:* Optimal scales ${k^* (x_i)}_(i=1)^n$ and estimates ${hat(f)(x_i)}_(i=1)^n$]
      LineBreak
      Comment[Initialization:]
      For($i = 1, dots, n$, {
        Assign[$L_0 (x_i)$][$-infinity$]
        Assign[$U_0 (x_i)$][$+infinity$]
        Assign[$k^* (x_i)$][$0$]
      })
      LineBreak
      Comment[For each scale $k = 1, 2, dots, K$:]
      For($i = 1, dots, n$, {
        Comment[(a) Compute LPA estimates $hat(f)_(h_k) (x_i)$ for all $i$]
        For($i = 1, dots, n$, {
          Assign[$hat(f)_(h_k) (x_i)$][LPA estimate with support $h_k$]
        })
        LineBreak
        Comment[(b) Compute variances $V(h_k, x_i) = sigma^2 sum_j h_(j,k)^2 (x_i)$]
        For($i = 1, dots, n$, {
          Assign[$V(h_k, x_i)$][$sigma^2 sum_j h_(j,k)^2 (x_i)$]
        })
        LineBreak
        Comment[(c) Calculate confidence bounds:]
        For($i = 1, dots, n$, {
          Assign[$l_k (x_i)$][$hat(f)_(h_k) (x_i) - gamma sqrt(V(h_k, x_i))$]
          Assign[$u_k (x_i)$][$hat(f)_(h_k) (x_i) + gamma sqrt(V(h_k, x_i))$]
        })
        LineBreak
        Comment[(d) Update intersection bounds:]
        For($i = 1, dots, n$, {
          Assign[$L_k (x_i)$][$max{L_(k-1) (x_i), l_k (x_i)}$]
          Assign[$U_k (x_i)$][$min{U_(k-1) (x_i), u_k (x_i)}$]
        })
        LineBreak
        Comment[(e) Check intersection validity:]
        For($i = 1, dots, n$, {
          If($U_k (x_i) < L_k (x_i) and k^* (x_i) = 0$, {
            Assign[$k^* (x_i)$][$k - 1$]
          })
        })
      })
      LineBreak
      Comment[Final estimates:]
      For($i = 1, dots, n$, {
        Assign[$hat(f)(x_i)$][$hat(f)_(h_(k^* (x_i))) (x_i)$]
      })
      Return[${k^* (x_i)}_(i=1)^n$, ${hat(f)(x_i)}_(i=1)^n$]
    },
  )
})

#pagebreak()
