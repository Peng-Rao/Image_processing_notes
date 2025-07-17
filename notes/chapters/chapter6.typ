#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "@preview/algorithmic:1.0.2"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm
#import "../template.typ": *

= Sparse Coding In $ell_1$ sense
== From $ell_0$ to $ell_1$ Regularization
The $ell_0$ norm, defined as the number of non-zero components in a vector, provides the most intuitive measure of sparsity. However, optimization problems involving the $ell_0$ norm are NP-hard due to their combinatorial nature. This computational intractability necessitates the exploration of alternative sparsity-promoting norms that maintain favorable optimization properties.

== The $ell_1$ Optimization Problem
The $ell_1$ optimization problem for sparse coding can be formulated in two equivalent ways:

#definition([Constrained $ell_1$ Problem (P1)])[
  $
    min_(bold(alpha)) norm(bold(alpha))_1 quad "subject to" quad norm(bold(D)bold(alpha) - bold(x))_2 <= epsilon
  $
]

#definition([Regularized $ell_1$ Problem (P2)])[
  $
    min_(bold(alpha)) 1/2 norm(bold(D)bold(alpha) - bold(b))_2^2 + lambda norm(bold(alpha))_1
  $
]

== Problem Components Analysis
=== Data Fidelity Term
The term $1/2 norm(bold(D)bold(alpha) - bold(x))_2^2$ serves as the data fidelity term, ensuring that the solution $bold(alpha)$ produces a reconstruction $bold(D)bold(alpha)$ that is close to the observed signal $bold(x)$.

#proposition([Properties of Data Fidelity Term])[
  The data fidelity term $g(bold(x)) = 1/2 norm(bold(D)bold(alpha) - bold(x))_2^2$ is:
  + Convex (as a composition of convex functions)
  + Differentiable with gradient $nabla g(bold(alpha)) = bold(D)^T (bold(D)bold(alpha) - bold(x))$
  + Strongly convex if $bold(D)$ has full column rank
]

=== Regularization Term
The term $lambda norm(bold(alpha))_1$ acts as a regularization term, promoting sparsity in the solution.

#proposition([Properties of $ell_1$ Regularization])[
  The regularization term $h(bold(alpha)) = norm(bold(alpha))_1$ is:
  + Convex
  + Non-differentiable at $alpha_i = 0$ for any component $i$
  + Promotes sparsity through its geometric properties
]

#pagebreak()

== Proximal Gradient Methods
=== Proximal Operator
For the composite optimization problem $min_(bold(alpha)) f(bold(alpha)) + g(bold(alpha))$, where $f$ is differentiable and $g$ is convex but not necessarily differentiable, we can use the proximal operator to handle the non-differentiable part.

#definition([Proximal Operator])[
  The proximal operator of a convex function $g$ is defined as:
  $
    "prox"_(lambda g)(bold(v)) = arg min_(bold(x)) {1/(2lambda) norm(bold(x) - bold(v))_2^2 + g(bold(x))}
  $
]

=== Proximal Gradient Descent
The proximal gradient descent algorithm iteratively updates the solution by combining the gradient step of the differentiable part and the proximal operator of the non-differentiable part.

For the problem $min_(bold(alpha)) f(bold(alpha)) + g(bold(alpha))$, the update rule is:
+ Initialize $bold(alpha)^(0)$
+ For $k = 0, 1, 2, dots$ until convergence:
  $
    bold(alpha)^(k+1) = "prox"_(t g)(bold(alpha)^(k) - t gradient f(bold(alpha)^(k)))
  $
  where $t$ is a step size parameter. Usually, $t$ is chosen based on the Lipschitz constant of $gradient f$. $t = 1/L$, where $L$ is the Lipschitz constant equal to the largest eigenvalue of $bold(D)^T bold(D)$.

=== Proximal Operator for $ell_1$ Norm
#theorem("Soft Thresholding")[
  The proximal operator of the $ell_1$ norm is given by the soft thresholding operator:

  $ ["prox"_(lambda norm(dot)_1)(bold(v))]_i = "sign"(v_i) max{|v_i| - lambda, 0} $

  where $"sign"(v_i)$ is the sign function.
]
*Proof:*
The proximal operator problem becomes:

$ min_(bold(x)) {1/(2lambda) norm(bold(x) - bold(v))_2^2 + norm(bold(x))_1} $

This separates into $n$ independent scalar problems:

$ min_(x_i) {1/(2lambda)(x_i - v_i)^2 + |x_i|} $

The solution is the soft thresholding operator due to the subdifferential analysis of the absolute value function.

=== Step Size Selection
#definition([Lipschitz Continuity of Gradient])[
  A function $f$ has $L$-Lipschitz continuous gradient if:
  $
    norm(nabla f(bold(x)) - nabla f(bold(y)))_2 <= L norm(bold(x) - bold(y))_2, quad forall bold(x), bold(y) in RR^n
  $
]

For the quadratic function $f(bold(x)) = 1/2 norm(bold(A)bold(x) - bold(b))_2^2$:

#lemma("Lipschitz Constant")[
  The Lipschitz constant of $nabla f$ is $L = norm(bold(A)^T bold(A))_2 = lambda_"max"(bold(A)^T bold(A))$, where $lambda_"max"$ denotes the largest eigenvalue.
]

*Proof: *
The gradient is $nabla f(bold(x)) = bold(A)^T (bold(A)bold(x) - bold(b))$. Thus:
$
  norm(nabla f(bold(x)) - nabla f(bold(y)))_2 & = norm(bold(A)^T bold(A)(bold(x) - bold(y)))_2              \
                                              & <= norm(bold(A)^T bold(A))_2 norm(bold(x) - bold(y))_2      \
                                              & = lambda_"max"(bold(A)^T bold(A)) norm(bold(x) - bold(y))_2
$
where we used the fact that the spectral norm equals the largest eigenvalue for symmetric positive semidefinite matrices.

=== Iterative Soft Thresholding Algorithm (ISTA)
#algorithm-figure("Iterative Soft Thresholding Algorithm (ISTA)", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([ISTA], ([$bold(D)$], [$bold(b)$], [$lambda$], [$epsilon$]), {
    Comment[*Input:*]
    Comment[$bold(D) in RR^(m times n)$: dictionary/measurement matrix]
    Comment[$bold(b) in RR^m$: observation vector]
    Comment[$lambda > 0$: regularization parameter]
    Comment[$epsilon > 0$: convergence tolerance]
    LineBreak
    Comment[*Output:*]
    Comment[$bold(alpha)^* in RR^n$: sparse coefficient vector]
    LineBreak
    Comment[Initialize]
    Assign[$bold(alpha)^(0)$][$bold(0)$]
    Assign[$L$][largest eigenvalue of $bold(D)^T bold(D)$]
    Assign[$t$][$1/L$]
    Assign[$k$][$0$]
    LineBreak
    While("not converged", {
      Comment[Gradient step]
      Assign[$bold(u)^(k)$][$bold(alpha)^(k) - t bold(D)^T (bold(D)bold(alpha)^(k) - bold(b))$]
      LineBreak
      Comment[Proximal step (soft thresholding)]
      For($i = 1 < n$, {
        Assign[$alpha_i^(k+1)$][$"sign"(u_i^(k)) dot max{|u_i^(k)| - lambda t, 0}$]
      })
      LineBreak
      Comment[Check convergence]
      If($||bold(alpha)^(k+1) - bold(alpha)^(k)||_2 < epsilon$, {
        Return[$bold(alpha)^(k+1)$]
      })
      Assign[$k$][$k + 1$]
    })
    Return[$bold(alpha)^(k)$]
  })
})

=== Backtracking Line Search
When computing eigenvalues is impractical, adaptive step size selection via backtracking provides a robust alternative:

#algorithm-figure(
  "Backtracking Line Search for ISTA",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      [Backtracking-Line-Search-ISTA],
      ([$bold(D)$], [$bold(b)$], [$lambda$], [$epsilon$]),
      {
        Comment[*Input:*]
        Comment[$bold(D) in RR^(m times n)$: dictionary/measurement matrix]
        Comment[$bold(b) in RR^m$: observation vector]
        Comment[$lambda > 0$: regularization parameter]
        Comment[$epsilon > 0$: convergence tolerance]
        Comment[*Parameters:* $beta in (0, 1)$ (typically $beta = 0.5$), $eta in (0, 1)$ (typically $eta = 0.9$)]
        LineBreak
        Comment[*Output:*]
        Comment[$bold(alpha)^* in RR^n$: sparse coefficient vector]
        LineBreak
        Comment[Initialize]
        Assign[$t$][$t_0$]
        Comment[initial step size, e.g., $t_0 = 1$]
        Assign[$bold(alpha)^(0)$][$bold(0)$]
        Assign[$k$][$0$]
        LineBreak
        While(
          "not converged",
          {
            Comment[Compute gradient step]
            Assign[$bold(g)^(k)$][$bold(D)^T (bold(D)bold(alpha)^(k) - bold(b))$]
            LineBreak
            Comment[Compute proximal operator (soft thresholding)]
            Assign[$bold(alpha)^+$][$"prox"_(t lambda ||dot||_1)(bold(alpha)^(k) - t bold(g)^(k))$]
            LineBreak
            Comment[Define objective function: $F(bold(alpha)) = 1/2 ||bold(D)bold(alpha) - bold(b)||_2^2 + lambda ||bold(alpha)||_1$]
            Comment[Backtracking line search]
            While($F(bold(alpha)^+) > F(bold(alpha)^(k)) - eta/t ||bold(alpha)^+ - bold(alpha)^(k)||_2^2$, {
              Comment[Decrease step size]
              Assign[$t$][$beta t$]
              Comment[Recompute proximal step with new step size]
              Assign[$bold(alpha)^+$][$"prox"_(t lambda ||dot||_1)(bold(alpha)^(k) - t bold(g)^(k))$]
            })
            LineBreak
            Comment[Check convergence]
            If($||bold(alpha)^+ - bold(alpha)^(k)||_2 < epsilon$, {
              Return[$bold(alpha)^+$]
            })
            Comment[Update for next iteration]
            Assign[$bold(alpha)^(k+1)$][$bold(alpha)^+$]
            Assign[$k$][$k + 1$]
          },
        )
        Return[$bold(alpha)^(k)$]
      },
    )
  },
)

#pagebreak()

== FISTA
While ISTA achieves $O(1/k)$ convergence, Nesterov's acceleration technique can improve this to $O(1/k^2)$ without additional computational cost per iteration. This acceleration is achieved through a momentum-like mechanism that exploits the history of iterates.

#algorithm-figure(
  "Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      [FISTA],
      ([$bold(D)$], [$bold(b)$], [$lambda$], [$epsilon$]),
      {
        Comment[*Input:*]
        Comment[$bold(D) in RR^(m times n)$: dictionary/measurement matrix]
        Comment[$bold(b) in RR^m$: observation vector]
        Comment[$lambda > 0$: regularization parameter]
        Comment[$epsilon > 0$: convergence tolerance]
        LineBreak
        Comment[*Output:*]
        Comment[$bold(alpha)^* in RR^n$: sparse coefficient vector]
        LineBreak
        Comment[Initialize]
        Assign[$bold(alpha)^0$][$bold(0)$]
        Assign[$bold(y)^1$][$bold(alpha)^0$]
        Assign[$t_1$][$1$]
        Assign[$L$][largest eigenvalue of $bold(D)^T bold(D)$]
        Assign[$alpha$][$1/L$]
        Assign[$k$][$1$]
        LineBreak
        While(
          "not converged",
          {
            Comment[(a) Proximal gradient step:]
            Assign[$bold(alpha)^k$][$"prox"_(alpha lambda norm(dot)_1)(bold(y)^k - alpha bold(D)^T (bold(D)bold(y)^k - bold(b)))$]
            LineBreak
            Comment[(b) Update momentum parameter:]
            Assign[$t_(k+1)$][$frac(1 + sqrt(1 + 4t_k^2), 2)$]
            LineBreak
            Comment[(c) Compute extrapolated point:]
            Assign[$bold(y)^(k+1)$][$bold(alpha)^k + frac(t_k - 1, t_(k+1))(bold(alpha)^k - bold(alpha)^(k-1))$]
            LineBreak
            Comment[Check convergence]
            If($norm(bold(alpha)^k - bold(alpha)^(k-1))_2 < epsilon$, {
              Return[$bold(alpha)^k$]
            })
            Assign[$k$][$k + 1$]
          },
        )
        Return[$bold(alpha)^k$]
      },
    )
  },
)

=== The Momentum Sequence
The sequence ${t_k}$ satisfies the recurrence relation:
$
  t_(k+1)^2 - t_(k+1) - t_k^2 = 0
$
This yields the closed-form expression:
$
  t_k = frac(k + 1, 2) + O(1) approx frac(k, 2) " for large " k
$

=== The Extrapolation Step
The extrapolation coefficient:
$
  beta_k = frac(t_k - 1, t_(k+1)) approx frac(k-2, k+1) arrow 1 " as " k arrow infinity
$
This creates an "overshoot" effect that accelerates convergence by anticipating the trajectory of the iterates.

=== Convergence Rate
#theorem("FISTA Convergence Rate")[
  For FISTA with step size $alpha = 1/L$, the following bound holds:
  $
    F(bold(x)^k) - F(bold(x)^*) <= frac(2L norm(bold(x)^0 - bold(x)^*)_2^2, (k+1)^2)
  $

]

The $O(1/k^2)$ rate is optimal for first-order methods on the class of convex functions with Lipschitz continuous gradients.

#pagebreak()
