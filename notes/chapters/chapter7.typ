#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *

= Structured Sparsity
The classical sparse coding framework seeks representations with minimal non-zero coefficients, treating each coefficient independently. However, in many practical applications, the _locations_ of non-zero coefficients exhibit inherent structure that standard sparsity models fail to exploit. Structured sparsity extends the sparse coding paradigm by incorporating prior knowledge about coefficient patterns, leading to more robust and interpretable representations.

Consider the fundamental *sparse coding problem*:
$ min_(bold(x)) 1/2 ||bold(y) - bold(D)bold(x)||_2^2 + lambda ||bold(x)||_0 $ <basic_sparse>

where $bold(y) in RR^m$ represents the observed signal, $bold(D) in RR^(m times n)$ denotes an overcomplete dictionary with $n > m$, and $bold(x) in RR^n$ contains the sparse coefficients.

This formulation promotes sparsity uniformly across all coefficients. However, structured sparsity recognizes that coefficients often exhibit dependencies or groupings that reflect the underlying data generation process.

+ *Multi-channel Signal Processing*: When processing multiple signals acquired from the same source (e.g., multi-electrode recordings, hyperspectral imaging), the active atoms in the dictionary tend to be consistent across channels.
+ *Texture Analysis*: Patches extracted from textured images share common structural elements, suggesting that their sparse representations should utilize similar dictionary atoms.
+ *Statistical Variable Selection*: In high-dimensional regression problems, predictors often form natural groups (e.g., dummy variables for categorical features, measurements from the same instrument).

#pagebreak()

== Joint Sparsity and Mixed Norms
=== Problem Formulation
Let $bold(Y) = [bold(y)_1, bold(y)_2, ..., bold(y)_L] in RR^(m times L)$ represent a collection of $L$ signals sharing common structural properties. The joint sparse coding problem seeks a coefficient matrix $bold(X) = [bold(x)_1, bold(x)_2, ..., bold(x)_L] in RR^(n times L)$ such that:

$ bold(Y) approx bold(D)bold(X) $ <joint_approx>

The key insight is that columns of $bold(X)$ should not only be individually sparse but should also share common support patterns.

=== Mixed Norms for Matrices
To enforce joint sparsity, we introduce the $(p,q)$-mixed norm for matrices.

#definition([$ell_(p,q)$ Mixed Norm])[
  For a matrix $bold(X) in RR^(n times L)$, the $(p,q)$-mixed norm is defined as:
  $ ||bold(X)||_(p,q) = (sum_(i=1)^n (sum_(j=1)^L |X_(i j)|^p)^(q/p))^(1/q) $ <mixed_norm_def>
]

This can be interpreted as:
$
  ||bold(X)||_(p,q) & = ||(||bold(x)^((1))||_p, ||bold(x)^((2))||_p, ..., ||bold(x)^((n))||_p)^T||_q \
                    & = ||bold(v)||_q
$

where $bold(x)^((i))$ denotes the $i$-th row of $bold(X)$, and $v_i = ||bold(x)^((i))||_p$.

*Frobenius Norm*: When $p = q = 2$:
$
  ||bold(X)||_(2,2) = sqrt(sum_(i,j) |X_(i j)|^2) = ||bold(X)||_F
$
*Entry-wise Norms*: When $p = q$, the mixed norm reduces to the vectorized $ell_p$ norm:
$
  ||bold(X)||_(p,p) = ||"vec"(bold(X))||_p
$
*Joint Sparsity Norm*: The $ell_(2,1)$ norm:
$
  ||bold(X)||_(2,1) = sum_(i=1)^n sqrt(sum_(j=1)^L |X_(i j)|^2) = sum_(i=1)^n ||bold(x)^((i))||_2
$ <l21_norm>

#proposition("Joint Sparsity Property")[
  The $ell_(2,1)$ norm promotes row-sparsity in $bold(X)$, meaning entire rows become zero. This corresponds to selecting the same dictionary atoms across all signals.
]

#pagebreak()

== Joint Sparse Coding
The joint sparse coding problem with the $ell_(2,1)$ norm regularization is formulated as:
$ min_(bold(X) in RR^(n times L)) 1/2 ||bold(Y) - bold(D)bold(X)||_F^2 + lambda ||bold(X)||_(2,1) $ <joint_sparse_opt>

This optimization problem combines a smooth, convex data fidelity term with a non-smooth but convex regularizer, making it amenable to proximal gradient methods.

The iterative solution via proximal gradient descent follows:
$
  bold(X)^((k+1)) = "prox"_(gamma lambda ||dot||_(2,1)) (bold(X)^((k)) - gamma nabla f(bold(X)^((k))))
$ <prox_grad_update>

where $f(bold(X)) = 1/2 ||bold(Y) - bold(D)bold(X)||_F^2$ is the smooth part, and $gamma > 0$ is the step size.

The key computational challenge lies in evaluating the proximal mapping:
$
  prox_(tau ||dot||_(2,1)) (bold(Z)) = argmin_(bold(X)) {tau ||bold(X)||_(2,1) + 1/2 ||bold(X) - bold(Z)||_F^2}
$ <prox_l21_def>

#theorem([Row-wise Separability of $ell_(2,1)$ Proximal Mapping])[
  The proximal mapping of the $ell_(2,1)$ norm can be computed row-wise:
  $ [prox_(tau ||dot||_(2,1)) (bold(Z))]_i = shrink_tau^((2)) (bold(z)^((i))) $
  where $bold(z)^((i))$ is the $i$-th row of $bold(Z)$, and $shrink_tau^((2))$ is the multivariate soft-thresholding operator.
]
*Proof:*
The objective function in @prox_l21_def can be rewritten as:
$
  tau ||bold(X)||_(2,1) + 1/2 ||bold(X) - bold(Z)||_F^2 &= tau sum_(i=1)^n ||bold(x)^((i))||_2 + 1/2 sum_(i=1)^n ||bold(x)^((i)) - bold(z)^((i))||_2^2 \
  &= sum_(i=1)^n [tau ||bold(x)^((i))||_2 + 1/2 ||bold(x)^((i)) - bold(z)^((i))||_2^2]
$
Since the objective decomposes into independent row-wise problems, the minimization can be performed separately for each row.

#definition("Multivariate Soft-Thresholding")[
  For a vector $bold(v) in RR^L$ and threshold $tau > 0$:
  $
    shrink_tau^((2)) (bold(v)) = cases(
      bold(v) / ||bold(v)||_2 dot max(0, ||bold(v)||_2 - tau) & "if " bold(v) != bold(0),
      bold(0) & "if " bold(v) = bold(0)
    )
  $ <multi_soft_thresh>
]

This operator exhibits two key behaviors:
1. *Nullification*: If $||bold(v)||_2 <= tau$, the entire vector is set to zero.
2. *Shrinkage*: If $||bold(v)||_2 > tau$, the vector is scaled down while preserving its direction.

#pagebreak()

== Group Sparsity and Extensions
Consider a dictionary $bold(D)$ partitioned into $G$ groups:
$
  bold(D) = [bold(D)_1 | bold(D)_2 | dots | bold(D)_G]
$
where $bold(D)_g in RR^(m times n_g)$ contains atoms corresponding to group $g$, and $sum_(g=1)^G n_g = n$.

The coefficient vector $bold(x)$ is correspondingly partitioned:
$
  bold(x) = mat(bold(x)_([1]); bold(x)_([2]); dots.v; bold(x)_([G]))
$
where $bold(x)_([g]) in RR^(n_g)$ contains coefficients for group $g$.

The group sparse coding problem seeks representations that activate entire groups rather than individual atoms:
$ min_(bold(x)) 1/2 ||bold(y) - bold(D)bold(x)||_2^2 + lambda sum_(g=1)^G w_g ||bold(x)_([g])||_2 $ <group_lasso>

where $w_g > 0$ are *group-specific weights*, typically set as $w_g = sqrt(n_g)$ to account for group size differences.

#attention("Group Selection Property")[
  The group LASSO penalty induces sparsity at the group level: either all coefficients within a group are zero, or the group is active with potentially multiple non-zero coefficients.
]

The proximal mapping for the group LASSO penalty decomposes into group-wise operations:
$
  [prox_(tau sum_g w_g ||dot||_2) (bold(z))]_([g]) = shrink_(tau w_g)^((2)) (bold(z)_([g]))
$
This allows efficient computation by applying multivariate soft-thresholding to each group independently.

#pagebreak()

== LASSO
In the statistical setting, we observe $m$ samples with response variable $y_i$ and $n$ predictors $bold(x)_i = [x_(i 1), x_(i 2), ..., x_(i n)]^T$. The linear model assumes:
$
  y_i = bold(x)_i^T bold(beta) + epsilon_i, quad i = 1, 2, ..., m
$ <linear_model>
In matrix notation:
$
  bold(y) = bold(X) bold(beta) + bold(epsilon)
$
where $bold(y) in RR^m$, $bold(X) in RR^(m times n)$ is the design matrix, $bold(beta) in RR^n$ contains regression coefficients, and $bold(epsilon) tilde cal(N)(bold(0), sigma^2 bold(I))$.

The classical least squares estimator:
$
  hat(bold(beta))_("LS") = argmin_(bold(beta)) ||bold(y) - bold(X)bold(beta)||_2^2 = (bold(X)^T bold(X))^(-1) bold(X)^T bold(y)
$ <ols>

This estimator is unbiased ($EE[hat(bold(beta))_("LS")] = bold(beta)$) but may have high variance, especially when predictors are correlated or $n$ is large relative to $m$.

The Least Absolute Shrinkage and Selection Operator (LASSO) adds $ell_1$ regularization:
$
  hat(bold(beta))_("LASSO") = argmin_(bold(beta)) {1/2 ||bold(y) - bold(X)bold(beta)||_2^2 + lambda ||bold(beta)||_1}
$ <lasso>

#theorem("Variable Selection Property")[
  For sufficiently large $lambda$, the LASSO estimator $hat(bold(beta))_("LASSO")$ contains exact zeros, performing automatic variable selection.
]

The LASSO introduces bias to reduce variance:
$
  "MSE"(hat(bold(beta))) & = EE[||hat(bold(beta)) - bold(beta)||_2^2]                               \
                         & = ||EE[hat(bold(beta))] - bold(beta)||_2^2 + trace(Var(hat(bold(beta)))) \
                         & = "Bias"^2 + "Variance"
$
While OLS minimizes bias, LASSO accepts some bias in exchange for substantially reduced variance through sparsity.

#theorem("LASSO in High Dimensions")[
  When $n > m$ (more predictors than observations), OLS is not unique. However, LASSO provides a unique sparse solution for appropriate $lambda > 0$.
]

This property makes LASSO particularly valuable in modern applications like genomics, where the number of features vastly exceeds the sample size.

#pagebreak()

== Elastic Net
The LASSO has two notable limitations:
1. In the $n > m$ setting, it selects at most $m$ variables
2. When predictors are highly correlated, LASSO tends to arbitrarily select one from each group

The Elastic Net addresses these issues by combining $ell_1$ and $ell_2$ penalties:
$
  hat(bold(beta))_("EN") = argmin_(bold(beta)) {1/2 ||bold(y) - bold(X)bold(beta)||_2^2 + lambda_1 ||bold(beta)||_1 + lambda_2 ||bold(beta)||_2^2}
$ <elastic_net>

=== Geometric Interpretation
The constraint region for Elastic Net is:
$
  cal(C)_("EN") = {bold(beta) : alpha ||bold(beta)||_1 + (1-alpha) ||bold(beta)||_2^2 <= t}
$
This creates a compromise between the diamond-shaped $ell_1$ ball and the spherical $ell_2$ ball, maintaining sparsity-inducing corners while allowing smoother boundaries.

=== Proximal Gradient Solution
The Elastic Net optimization can be solved efficiently using proximal gradient methods. The key insight is that the $ell_2$ penalty can be absorbed into the smooth part:
$ f(bold(beta)) = 1/2 ||bold(y) - bold(X)bold(beta)||_2^2 + lambda_2 ||bold(beta)||_2^2 $
with gradient:
$
  nabla f(bold(beta)) = bold(X)^T (bold(X)bold(beta) - bold(y)) + 2 lambda_2 bold(beta) = (bold(X)^T bold(X) + 2 lambda_2 bold(I)) bold(beta) - bold(X)^T bold(y)
$
The proximal gradient update becomes:
$
  bold(beta)^((k+1)) = shrink_(gamma lambda_1) (bold(beta)^((k)) - gamma nabla f(bold(beta)^((k))))
$
where $shrink_tau$ is the element-wise soft-thresholding operator.

== Algorithmic Summary
#figure(
  table(
    columns: 3,
    align: left,
    table.header([*Problem*], [*Regularizer*], [*Proximal Operator*]),
    [Standard Sparsity], [$||bold(x)||_1$], [$shrink_tau (x_i) = sign(x_i) max(0, |x_i| - tau)$],
    [Joint Sparsity], [$||bold(X)||_(2,1)$], [Row-wise multivariate soft-thresholding],
    [Group Sparsity], [$sum_g w_g ||bold(x)_([g])||_2$], [Group-wise multivariate soft-thresholding],
    [Elastic Net],
    [$lambda_1 ||bold(x)||_1 + lambda_2 ||bold(x)||_2^2$],
    [Modified soft-thresholding with $ell_2$ in gradient],
  ),
  caption: [Summary of sparsity-inducing regularizers and their proximal operators],
)

#pagebreak()
