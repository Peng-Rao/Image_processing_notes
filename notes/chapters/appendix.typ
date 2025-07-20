#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *
= Appendix: Fundamental Concepts in Linear Algebra
== Vector Spaces and Linear Combinations
#definition("Span of Vectors")[
  Given a set of vectors ${bold(v)_1, bold(v)_2, ..., bold(v)_n} subset RR^m$, the _span_ of these vectors is defined as:

  $ "span"{bold(v)_1, bold(v)_2, ..., bold(v)_n} = {sum_(i=1)^n lambda_i bold(v)_i : lambda_i in RR} $
]

The span represents the set of all possible linear combinations of the given vectors, forming a vector subspace of $RR^m$. This concept is fundamental to understanding how different sets of vectors can generate different subspaces.

== Linear Independence and Basis
#definition("Linear Independence")[
  A set of vectors ${bold(v)_1, bold(v)_2, ..., bold(v)_n}$ is _linearly independent_ if and only if:

  $ sum_(i=1)^n lambda_i bold(v)_i = bold(0) quad => quad lambda_i = 0 " for all " i = 1, 2, ..., n $
]

This definition captures the fundamental property that no vector in the set can be expressed as a linear combination of the others. The importance of linear independence becomes clear when we consider the uniqueness of representations.

#theorem("Uniqueness of Representation")[
  Let ${bold(e)_1, bold(e)_2, ..., bold(e)_n} subset RR^m$ be a linearly independent set of vectors. If $bold(s) in "span"{bold(e)_1, bold(e)_2, ..., bold(e)_n}$, then there exists a unique representation:

  $ bold(s) = sum_(i=1)^n x_i bold(e)_i $

  where the coefficients $x_i in RR$ are uniquely determined.
] <thm:uniqueness>

*Proof: *
Suppose $bold(s)$ admits two different representations:
$
  bold(s) & = sum_(i=1)^n x_i bold(e)_i \
  bold(s) & = sum_(i=1)^n y_i bold(e)_i
$
Subtracting these equations:
$
  bold(0) = sum_(i=1)^n (x_i - y_i) bold(e)_i
$
By linear independence, $(x_i - y_i) = 0$ for all $i$, implying $x_i = y_i$ for all $i$. Therefore, the representation is unique.

#pagebreak()

== Orthogonal Vectors
#definition("Orthonormal Basis")[
  A set of vectors ${bold(e)_1, bold(e)_2, ..., bold(e)_n} subset RR^n$ forms an _orthonormal basis_ if:
  + $angle.l bold(e)_i, bold(e)_j angle.r = delta_(i j)$ (orthonormality condition)
  + $"span"{bold(e)_1, bold(e)_2, ..., bold(e)_n} = RR^n$ (spanning condition)
  where $delta_(i j)$ is the Kronecker delta.
  Kronecker delta is defined as:
  $
    delta_(i j) = cases(
      1 space "if" i = j,
      0 space "if" i != j
    )
  $
]

The power of orthonormal bases lies in their computational convenience. For any signal $bold(s) in RR^n$ and orthonormal basis $bold(D) = [bold(e)_1, bold(e)_2, ..., bold(e)_n]$, the coefficient computation is straightforward:
$
  bold(x) = bold(D)^T bold(s)
$
where $bold(x) = (x_1, x_2, ..., x_n)^T$ and $x_i = angle.l bold(e)_i, bold(s) angle.r$.

== The $ell_0$ Norm
The $ell_0$ norm (more precisely, $ell_0$ pseudo-norm) of a vector $bold(x) in RR^n$ is defined as:
$
  norm(bold(x))_0 := |{i : x_i != 0}| = sum_(i=1)^n bold(1)_(x_i != 0)
$
where $bold(1)_{x_i != 0}$ is the indicator function that equals 1 if $x_i != 0$ and 0 otherwise.

The $ell_0$ norm can be understood as the limit of $ell_p$ norms as $p -> 0^+$:
$
  norm(bold(x))_0 = lim_(p -> 0^+) norm(bold(x))_p^p = lim_(p -> 0^+) (sum_(i=1)^n |x_i|^p)
$


The $ell_0$ norm satisfies the following properties:
+ *Non-negativity*: $norm(bold(x))_0 >= 0$ for all $bold(x) in RR^n$
+ *Zero property*: $norm(bold(x))_0 = 0$ if and only if $bold(x) = bold(0)$
+ *Triangle inequality*: $norm(bold(x) + bold(y))_0 <= norm(bold(x))_0 + norm(bold(y))_0$
+ *Failure of homogeneity*: $norm(lambda bold(x))_0 != |lambda| norm(bold(x))_0$ for $lambda != 0, plus.minus 1$


Let us now interpret $ell_0$-sparsity geometrically in $RR^3$.
- $norm(bold(alpha))_0 = 0$: Only the origin $(0, 0, 0)$
- $norm(bold(alpha))_0 = 1$: Points on coordinate axes, e.g., $(7, 0, 0)$, $(0, 3, 0)$
- $norm(bold(alpha))_0 = 2$: Points lying in coordinate planes, e.g., $(5, 2, 0)$
- $norm(bold(alpha))_0 = 3$: All other points in $RR^3$

#pagebreak()

== Matrix Spark <matrix-spark>
The concept of matrix spark provides the theoretical foundation for understanding when sparse solutions are unique.

#definition("Matrix Spark")[
  For a matrix $bold(D) in RR^(m times n)$, the spark of $bold(D)$, denoted $"spark"(bold(D))$, is defined as:
  $
    "spark"(bold(D)) = min{|cal(S)| : cal(S) subset.eq {1, 2, ..., n}, bold(D)_cal(S) "is linearly dependent"}
  $
  where $|cal(S)|$ denotes the cardinality of the set $cal(S)$ and $bold(D)_cal(S)$ represents the submatrix of $bold(D)$ formed by columns indexed by $cal(S)$.

  Equivalently, the spark can be defined in terms of the $ell_0$ norm as:
  $
    "spark"(bold(D)) = min{norm(bold(x))_0 : bold(D) bold(x) = bold(0), bold(x) != bold(0)}
  $
] <def:spark>

The spark and rank of a matrix are related but distinct concepts:

#proposition("Spark-Rank Relationship")[
  For any matrix $bold(D) in RR^(m times n)$ with $"rank"(bold(D)) = r$:
  $
    1 <= "spark"(bold(D)) <= r + 1
  $
] <prop:spark_rank>

*Proof:*
The lower bound follows from the definition. For the upper bound, consider that any $r+1$ columns must be linearly dependent in an $r$-dimensional space, hence $"spark"(bold(D)) <= r + 1$.

#lemma("Spark and Homogeneous Solutions")[
  If $bold(D) bold(x) = bold(0)$ has a solution $bold(x) != bold(0)$, then:
  $
    "spark"(bold(D)) <= norm(bold(x))_0
  $
] <lem:spark_homogeneous>

*Proof:*
If $bold(D) bold(x) = bold(0)$ with $bold(x) != bold(0)$, then $sum_(i in "supp"(bold(x))) x_i bold(d)_i = bold(0)$, showing that the columns ${bold(d)_i : i in "supp"(bold(x))}$ are linearly dependent. By definition of spark, $"spark"(bold(D)) <= |"supp"(bold(x))| = norm(bold(x))_0$.

#pagebreak()

= Appendix: Optimization Theory for Non-Differentiable Functions
== Lipschitz Continuity <lipschitz-continuity>
#definition("Lipschitz Continuous Gradient")[
  A function $f: RR^n -> RR$ has an $L$-*Lipschitz continuous gradient* if there exists a constant $L > 0$ such that:
  $
    norm(nabla f(bold(x)) - nabla f(bold(y)))_2 <= L norm(bold(x) - bold(y))_2
  $
  for all $bold(x), bold(y) in RR^n$. The smallest such constant $L$ is called the Lipschitz constant of the gradient.
]

#definition([Lipschitz Continuity of Gradient])[
  A function $f$ has $L$-Lipschitz continuous gradient if:
  $
    norm(nabla f(bold(x)) - nabla f(bold(y)))_2 <= L norm(bold(x) - bold(y))_2, quad forall bold(x), bold(y) in RR^n
  $
]

For the quadratic function $f(bold(x)) = 1/2 norm(bold(A)bold(x) - bold(b))_2^2$:

#proposition("Lipschitz Constant for Quadratic Functions")[
  For the quadratic function $f(bold(x)) = 1/2 norm(bold(A)bold(x) - bold(b))_2^2$, the Lipschitz constant can be derived as follows:
  $
    L = norm(bold(A)^T bold(A))_2 = lambda_"max"(bold(A)^T bold(A))
  $
  where $lambda_"max"$ denotes the largest eigenvalue.

  *Proof: *
  The gradient is $nabla f(bold(x)) = bold(A)^T (bold(A)bold(x) - bold(b))$. Thus:
  $
    norm(nabla f(bold(x)) - nabla f(bold(y)))_2 & = norm(bold(A)^T bold(A)(bold(x) - bold(y)))_2              \
                                                & <= norm(bold(A)^T bold(A))_2 norm(bold(x) - bold(y))_2      \
                                                & = lambda_"max"(bold(A)^T bold(A)) norm(bold(x) - bold(y))_2
  $
  where we used the fact that the spectral norm equals the largest eigenvalue for symmetric positive semidefinite matrices.
]

#pagebreak()

== Majorization-Minimization <majorization-minimization>
For smooth convex functions, we can construct quadratic majorizers that facilitate optimization (subgradient methods).

#lemma("Descent Lemma")[
  Let $f: RR^n -> RR$ be a convex, differentiable function with Lipschitz continuous gradient. Then for any $bold(x)_k in RR^n$, there exists $L > 0$ such that:
  $
    f(bold(x)) <= Q_L (bold(x); bold(x)_k) = f(bold(x)_k) + gradient f(bold(x)_k)^T (bold(x) - bold(x)_k) + L/2 norm(bold(x) - bold(x)_k)_2^2
  $
  for all $bold(x) in RR^n$.
]

#definition("Majorization-Minimization Algorithm")[
  Given a convex function $f$, the majorization-minimization approach generates a sequence ${bold(x)_k}$ by:
  $
    bold(x)_(k+1) &= arg min_(bold(x)) Q_L (bold(x); bold(x)_k) \
    &= arg min_(bold(x)) [f(bold(x)_k) + gradient f(bold(x)_k)^T (bold(x) - bold(x)_k) + L/2 norm(bold(x) - bold(x)_k)_2^2]
  $
]

Minimizing the majorizer $Q_L (bold(x); bold(x)_k)$ with respect to $bold(x)$:
$
  gradient_(bold(x)) Q_L (bold(x); bold(x)_k) & = gradient f(bold(x)_k) + L(bold(x) - bold(x)_k) = bold(0) \
                             => bold(x)_(k+1) & = bold(x)_k - 1/L gradient f(bold(x)_k)
$
This recovers the standard gradient descent update with step size $gamma = 1/L$.

For a convex, differentiable function $f$ with Lipschitz continuous gradient, the gradient descent algorithm converges to the global minimum.
