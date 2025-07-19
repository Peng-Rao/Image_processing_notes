#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *

= Introduction to Dictionary Learning
The solution to the sparsity limitation lies in abandoning the constraint of orthonormality and embracing redundancy. Instead of using a single $n times n$ orthonormal basis, we construct an $n times m$ dictionary matrix $bold(D)$ where $m > n$.

#definition("Overcomplete Dictionary")[
  An _overcomplete dictionary_ is a matrix $bold(D) in RR^(n times m)$ with $m > n$ such that:
  $
    "span"{bold(d)_1, bold(d)_2, ..., bold(d)_m} = RR^n
  $
  where $bold(d)_i$ are the columns of $bold(D)$.
]

For the DCT-spike example, we construct the overcomplete dictionary by concatenating the DCT basis with the *canonical basis*:

$
  bold(D) = mat(bold(D)_"DCT", bold(I)) in RR^(n times 2n)
$

This construction ensures that:
- Signals sparse in DCT domain remain sparse
- Signals sparse in canonical domain remain sparse
- Mixed signals (DCT-sparse + spikes) admit sparse representations

#example("Sparse Representation with Overcomplete Dictionary")[
  Consider the signal $bold(s) = bold(s)_0 + lambda bold(e)_j$ where $bold(s)_0 = bold(D)_"DCT" bold(x)_0$ with sparse $bold(x)_0$.

  The representation with respect to the overcomplete dictionary is:

  $
    bold(s) = bold(D) mat(bold(x)_0; lambda bold(e)_j)
  $

  The coefficient vector $mat(bold(x)_0; lambda bold(e)_j) in RR^(2n)$ is sparse, containing only the non-zero entries of $bold(x)_0$ plus the single entry $lambda$ at position $j$ in the second block.
]

== Theoretical Properties of Overcomplete Systems
#theorem("Rouché-Capelli Theorem")[
  Consider the linear system $bold(D)bold(x) = bold(s)$ where $bold(D) in RR^(n times m)$ and $bold(s) in RR^n$. The system admits a solution if and only if:

  $
    "rank"(bold(D)) = "rank"(mat(bold(D), bold(s)))
  $
] <thm:rouche>

When $m > n$ and $"rank"(bold(D)) = n$, the system has infinitely many solutions forming an affine subspace of dimension $m - n$.

#theorem("Solution Space Dimension")[
  If $bold(D) in RR^(n times m)$ with $m > n$ and $"rank"(bold(D)) = n$, then for any $bold(s) in RR^n$, the solution set of $bold(D)bold(x) = bold(s)$ forms an affine subspace of dimension $m - n$.
]

The abundance of solutions in overcomplete systems necessitates additional criteria for solution selection. This is where *regularization theory* becomes essential.

#pagebreak()

== Regularization and Sparse Recovery
#definition("Regularization")[
  Given an *ill-posed* problem $bold(D)bold(x) = bold(s)$ with multiple solutions, _regularization_ involves solving:
  $
    hat(bold(x)) = arg min_(bold(x)) J(bold(x)) quad "subject to" quad bold(D)bold(x) = bold(s)
  $
  where $J: RR^m -> RR_+$ is a regularization functional encoding our prior knowledge about the desired solution.
]

=== $ell_2$ Regularization: Ridge Regression
The most mathematically tractable regularization is the $ell_2$ norm:
$
  J(bold(x)) = 1/2 ||bold(x)||_2^2 = 1/2 sum_(i=1)^m x_i^2
$
This leads to the constrained optimization problem:
$
  hat(bold(x)) = arg min_(bold(x)) 1/2 ||bold(x)||_2^2 quad "subject to" quad bold(D)bold(x) = bold(s)
$
Alternatively, we can formulate the unconstrained version:
$
  hat(bold(x)) = arg min_(bold(x)) 1/2 ||bold(D)bold(x) - bold(s)||_2^2 + lambda/2 ||bold(x)||_2^2
$

#theorem("Ridge Regression Solution")[
  The solution to the ridge regression problem:

  $ hat(bold(x)) = arg min_(bold(x)) 1/2 ||bold(D)bold(x) - bold(s)||_2^2 + lambda/2 ||bold(x)||_2^2 $

  is given by:

  $ hat(bold(x)) = (bold(D)^T bold(D) + lambda bold(I))^(-1) bold(D)^T bold(s) $

  where $lambda > 0$ ensures the matrix $(bold(D)^T bold(D) + lambda bold(I))$ is invertible.

  *Proof*:
  Define the objective function:

  $ f(bold(x)) = 1/2 ||bold(D)bold(x) - bold(s)||_2^2 + lambda/2 ||bold(x)||_2^2 $

  Expanding the squared norms:

  $
    f(bold(x)) &= 1/2 (bold(D)bold(x) - bold(s))^T (bold(D)bold(x) - bold(s)) + lambda/2 bold(x)^T bold(x) \
    &= 1/2 bold(x)^T bold(D)^T bold(D) bold(x) - bold(D) bold(x)^T bold(s) + 1/2 bold(s)^T bold(s) + lambda/2 bold(x)^T bold(x)
  $

  Taking the gradient with respect to $bold(x)$:

  $ nabla f(bold(x)) = bold(D)^T bold(D) bold(x) - bold(D)^T bold(s) + lambda bold(x) $

  Setting $nabla f(bold(x)) = bold(0)$:

  $ (bold(D)^T bold(D) + lambda bold(I)) bold(x) = bold(D)^T bold(s) $

  Since $lambda > 0$, the matrix $(bold(D)^T bold(D) + lambda bold(I))$ is positive definite and therefore invertible, yielding the stated solution.
] <thm:ridge>

=== Limitations of $ell_2$ Regularization
While $ell_2$ regularization provides a computationally efficient solution, it does not promote sparsity. The solution $hat(bold(x))$ typically has all non-zero entries, which contradicts our goal of sparse representation.

#attention([Sparsity vs. $ell_2$ Regularization])[
  The $ell_2$ norm penalizes large coefficients but does not drive small coefficients to zero. For sparse recovery, we need regularization functionals that promote sparsity, such as the $ell_1$ norm or $ell_0$ pseudo-norm.
]

== Towards Sparsity: $ell_0$ and $ell_1$ Regularization
=== The $ell_0$ "Norm" and True Sparsity
The most natural regularization for sparse recovery is the $ell_0$ "norm" (technically a pseudo-norm):

$ ||bold(x)||_0 = |{i : x_i != 0}| $

This counts the number of non-zero entries in $bold(x)$. The corresponding optimization problem:

$ hat(bold(x)) = arg min_(bold(x)) ||bold(x)||_0 quad "subject to" quad bold(D)bold(x) = bold(s) $

directly seeks the sparsest representation.

=== Computational Challenges
The $ell_0$ minimization problem is NP-hard in general, making it computationally intractable for large-scale problems. This has led to the development of convex relaxations and approximation algorithms.

#pagebreak()
