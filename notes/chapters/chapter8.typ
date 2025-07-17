#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *

= Dictionary Learning
== Introduction to Dictionary Learning
#definition("Dictionary Learning Problem")[
  Given a set of training signals $bold(y)_1, bold(y)_2, ..., bold(y)_N in RR^n$, the dictionary learning problem seeks to find:
  + A dictionary matrix $bold(D) in RR^{n times m}$ with $m > n$ (redundant dictionary)
  + Sparse coefficient vectors $bold(x)_1, bold(x)_2, ..., bold(x)_N in RR^m$

  such that $bold(y)_i approx bold(D) bold(x)_i$ for all $i = 1, 2, ..., N$, where each $bold(x)_i$ has at most $T_0$ non-zero entries.
]

Dictionary learning employs a block coordinate descent strategy, alternating between two phases:
+ *Sparse Coding Phase:* Fix $D$ and solve for $X$
+ *Dictionary Update Phase:* Fix $X$ and update $D$

== Problem Formulation
Let $bold(Y) = [bold(y)_1, bold(y)_2, ..., bold(y)_N] in RR^(n times N)$ denote the training matrix, where each column represents a training signal. Similarly, let $bold(X) = [bold(x)_1, bold(x)_2, ..., bold(x)_N] in RR^(m times N)$ represent the sparse coefficient matrix.

The dictionary learning problem can be formulated as the following optimization:
$
  min_(bold(D), bold(X)) norm(bold(Y) - bold(D) bold(X))_F^2
  quad "subject to" quad norm(bold(x)_i)_0 <= T_0, quad forall i = 1, 2, ..., N
$ <eq:dictionary_learning_problem>
where $norm(dot)_F$ denotes the Frobenius norm and $norm(dot)_0$ is the $ell_0$ pseudo-norm counting non-zero entries.


*(Normalization Constraint:)*. To resolve scaling ambiguities, we impose the constraint that each column of $bold(D)$ has unit $ell_2$ norm:
$
  norm(bold(d)_j)_2 = 1, quad forall j = 1, 2, ..., m
$

The optimization problem @eq:dictionary_learning_problem presents several fundamental challenges:
+ *Non-convexity*: The objective function is non-convex in the joint variables $bold(D), bold(X)$, even though it is convex in each variable individually when the other is fixed.
+ *Sparsity Constraint*: The $ell_0$ pseudo-norm constraint is non-convex and combinatorial, making direct optimization intractable.
+ *Solution ambiguity*: Multiple equivalent solutions exist due to:
  - Column permutations of $bold(D)$ with corresponding row permutations of $bold(X)$
  - Sign ambiguities: $bold(d)_j, bold(x)_j) equiv (-bold(d)_j, -bold(x)_j)$

#pagebreak()

== Sparse Coding Phase
The sparse coding phase solves the following problem for each training signal:

$
  bold(x)_i^((k+1)) = arg min_(bold(x)) norm(bold(y)_i - bold(D)^((k)) bold(x))_2^2
  quad "subject to" quad norm(bold(x))_0 <= T_0
$ <eq:sparse_coding>

This is precisely the sparse coding problem discussed in previous sections, which can be solved using greedy algorithms such as:

- *Orthogonal Matching Pursuit (OMP)*: Iteratively selects atoms that best correlate with the current residual
- *Matching Pursuit (MP)*: Similar to OMP but without orthogonalization
- *Basis Pursuit*: Convex relaxation using $ell_1$ norm

The sparse coding phase is computationally the most expensive part of dictionary learning, as it requires solving $N$ sparse coding problems (one for each training signal) at each iteration.

== Dictionary Update Phase
The dictionary update phase constitutes the core innovation of K-SVD. Rather than updating the entire dictionary simultaneously, K-SVD updates one column at a time while simultaneously updating the corresponding sparse coefficients.

=== Matrix Factorization Perspective
Consider the error matrix:
$
  bold(E) = bold(Y) - bold(D) bold(X)
$ <eq:error_matrix>
Using the fundamental matrix identity, we can decompose the product $bold(D) bold(X)$ as:
$
  bold(D) bold(X) = sum_(j=1)^m bold(d)_j bold(x)_j^T
$ <eq:matrix_decomp>
where $bold(x)_j^T$ denotes the $j$-th row of $bold(X)$.

=== Isolated Column Update
To update the $j_0$-th column of $bold(D)$, we rewrite @eq:matrix_decomp as:
$
  bold(D) bold(X) = sum_(j != j_0) bold(d)_j bold(x)_j^T + bold(d)_(j_0) bold(x)_(j_0)^T
$ <eq:isolated_column>
Define the error matrix excluding the $j_0$-th atom:
$
  bold(E)_(j_0) = bold(Y) - sum_(j != j_0) bold(d)_j bold(x)_j^T
$ <eq:error_j0>
The update problem becomes:
$
  min_(bold(d)_(j_0), bold(x)_(j_0)^T) norm(bold(E)_(j_0) - bold(d)_(j_0) bold(x)_(j_0)^T)_F^2
  quad "subject to" quad norm(bold(d)_(j_0))_2 = 1
$ <eq:rank_one_approx>
This is a rank-one matrix approximation problem, optimally solved using the Singular Value Decomposition (SVD).

=== SVD solution
#theorem("Rank-One Matrix Approximation")[
  Let $bold(A) in RR^{n times N}$ be given. The solution to
  $
    min_(bold(u), bold(v)) norm(bold(A) - bold(u) bold(v)^T)_F^2 quad "subject to" quad norm(bold(u))_2 = 1
  $
  is given by $bold(u) = bold(u)_1$ and $bold(v)^T = sigma_1 bold(v)_1^T$, where $bold(A) = sum_(i=1)^(min(n, N)) sigma_i bold(u)_i bold(v)_i^T$ is the SVD of $bold(A)$.
] <thm:rank_one>

*Proof:*
The Frobenius norm can be expressed as:
$
  norm(bold(A) - bold(u) bold(v)^T)_F^2 &= norm(bold(A))_F^2 - 2 "trace"(bold(A)^T bold(u) bold(v)^T) + norm(bold(u) bold(v)^T)_F^2 \
  &= norm(bold(A))_F^2 - 2 bold(v)^T bold(A)^T bold(u) + norm(bold(v))_2^2
$

Since $norm(bold(u))_2 = 1$, maximizing $bold(v)^T bold(A)^T bold(u)$ is equivalent to finding the leading singular vectors of $bold(A)$.

=== Sparsity Preservation
A critical challenge in the dictionary update is preserving the sparsity structure of $bold(X)$. The naive application of @thm:rank_one would yield a dense row vector $bold(x)_(j_0)^T$, violating the sparse coding constraint.

*Solution - Restricted SVD:* K-SVD addresses this by restricting the update to only those training signals that actually use the $j_0$-th atom:
$
  Omega_(j_0) = {i : x_(j_0,i) != 0}
$ <eq:support_set>
Define the restricted error matrix:
$
  bold(E)_(j_0)^R = bold(E)_(j_0)(:, Omega_(j_0))
$ <eq:restricted_error>
The restricted update problem becomes:
$
  min_(bold(d)_(j_0), bold(x)_(j_0)^R) norm(bold(E)_(j_0)^R - bold(d)_(j_0) (bold(x)_(j_0)^R)^T)_F^2
  quad "subject to" quad norm(bold(d)_(j_0))_2 = 1
$ <eq:restricted_update>
where $bold(x)_(j_0)^R$ contains only the non-zero elements of $bold(x)_(j_0)^T$.

*Dictionary Update Algorithm for Column $j_0$:*
+ Compute error matrix: $bold(E)_(j_0) = bold(Y) - sum_(j != j_0) bold(d)_j bold(x)_j^T$
+ Identify support: $Omega_(j_0) = {i : x_(j_0,i) != 0}$
+ Extract restricted matrix: $bold(E)_(j_0)^R = bold(E)_(j_0)(:, Omega_(j_0))$
+ Compute SVD: $bold(E)_(j_0)^R = bold(U) bold(Sigma) bold(V)^T$
+ Update dictionary: $bold(d)_(j_0) = bold(u)_1$
+ Update coefficients: $bold(x)_(j_0)^R = sigma_1 bold(v)_1$
+ Restore to full representation: $bold(x)_(j_0)^T (Omega_(j_0)) = bold(x)_(j_0)^R$

#pagebreak()
