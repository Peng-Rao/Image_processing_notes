#import "@local/simple-note:0.0.1": *
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

$ bold(0) = sum_(i=1)^n (x_i - y_i) bold(e)_i $

By linear independence, $(x_i - y_i) = 0$ for all $i$, implying $x_i = y_i$ for all $i$. Therefore, the representation is unique.

== Orthogonal Vectors
#definition("Orthonormal Basis")[
  A set of vectors ${bold(e)_1, bold(e)_2, ..., bold(e)_n} subset RR^n$ forms an _orthonormal basis_ if:

  + $angle.l bold(e)_i, bold(e)_j angle.r = delta_(i j)$ (orthonormality condition)
  + $"span"{bold(e)_1, bold(e)_2, ..., bold(e)_n} = RR^n$ (spanning condition)

  where $delta_(i j)$ is the Kronecker delta.
]

The power of orthonormal bases lies in their computational convenience. For any signal $bold(s) in RR^n$ and orthonormal basis $bold(D) = [bold(e)_1, bold(e)_2, ..., bold(e)_n]$, the coefficient computation is straightforward:

$ bold(x) = bold(D)^T bold(s) $

where $bold(x) = (x_1, x_2, ..., x_n)^T$ and $x_i = angle.l bold(e)_i, bold(s) angle.r$.
