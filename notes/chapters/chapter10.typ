#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *
#import "@preview/algorithmic:1.0.2"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

= Robust Fitting
The problem of _robust model fitting_ arises ubiquitously in computer vision applications where we must estimate analytical expressions derived from geometric constraints in the presence of outliers. Unlike standard noise, outliers represent data points that significantly deviate from the expected model with statistical characteristics that are fundamentally different from the inlier distribution.

#definition("Robust Fitting Problem")[
  Given a set of observations $cal(D) = {(bold(x)_i, y_i)}_(i=1)^n$ where $bold(x)_i in RR^d$ and $y_i in RR$, the robust fitting problem seeks to estimate parameters $bold(theta) in RR^p$ of a model $f(bold(x); bold(theta))$ such that:
  $ bold(theta)^* = argmin_(bold(theta)) sum_(i=1)^n rho(r_i (bold(theta))) $
  where $r_i (bold(theta)) = y_i - f(bold(x)_i; bold(theta))$ is the residual and $rho: RR -> RR^+$ is a robust loss function.
]

The choice of $rho$ fundamentally determines the robustness properties of the estimator. Classical least squares uses $rho(r) = r^2$, which, as we shall demonstrate, lacks robustness to outliers.

#figure(
  canvas(length: 1cm, {
    import draw: *

    // Coordinate system
    line((-1, 0), (6, 0), mark: (end: ">"))
    content((6.2, 0), [$x$])
    line((0, -1), (0, 5), mark: (end: ">"))
    content((0, 5.2), [$y$])

    // Inliers (blue points)
    let inliers = ((0.5, 1.2), (1, 1.8), (1.5, 2.3), (2, 2.9), (2.5, 3.4), (3, 3.9), (3.5, 4.5), (4, 5.0))
    for point in inliers {
      circle(point, radius: 0.1, fill: blue)
    }

    // Outliers (red points)
    let outliers = ((1.5, 4.5), (3, 1.5), (4.5, 2.0))
    for point in outliers {
      circle(point, radius: 0.1, fill: red)
    }

    // True line (blue dashed)
    line((-0.5, 0.5), (4.5, 5.5), stroke: (paint: blue, dash: "dashed", thickness: 2pt))
    content((4.7, 5.5), [True model])

    // Least squares line (red)
    line((-0.5, 1.0), (4.5, 4.0), stroke: (paint: red, thickness: 2pt))
    content((4.7, 4.0), [LS fit])
  }),
  caption: [Illustration of outlier influence on least squares fitting],
)

== Limitations of Vertical Distance Minimization
We begin with the fundamental problem of fitting a straight line to a set of 2D points. The parametric form of a line is:
$
  y = m x + b
$
where $m$ is the slope and $b$ is the y-intercept. Given observations ${(x_i, y_i)}_(i=1)^n$, ordinary least squares (OLS) seeks:
$
  (m^*, b^*) = argmin_(m,b) sum_(i=1)^n (y_i - m x_i - b)^2
$

The ordinary least squares approach measures error along the vertical axis, which introduces several fundamental limitations:
+ *Inability to fit vertical lines*: When the line approaches vertical ($m -> infinity$), the formulation breaks down as we cannot express $x$ as a function of $y$.
+ *Asymmetric treatment of variables*: The choice of dependent vs. independent variable artificially privileges one coordinate axis.
+ *Scale dependency*: The error metric depends on the coordinate system orientation.

== The Direct Linear Transformation (DLT) Algorithm
To overcome the limitations of parametric representations, we adopt the implicit form:
$
  a x + b y + c = 0
$
where $(a, b, c) in RR^3$ are the line parameters. This representation elegantly handles all line orientations, including vertical lines (where $b = 0$).

#attention("Scale Ambiguity")[
  The implicit representation introduces a scale ambiguity: the lines defined by $(a, b, c)$ and $(lambda a, lambda b, lambda c)$ for any $lambda != 0$ are identical. This necessitates a normalization constraint.
]

=== Algebraic Error Minimization
Rather than minimizing geometric distance (which leads to nonlinear optimization), we minimize the _algebraic error_:

#definition("Algebraic Error")[
  For a point $(x_i, y_i)$ and line parameters $bold(theta) = [a, b, c]^T$, the algebraic error is:
  $ r_i^"alg"(bold(theta)) = a x_i + b y_i + c $
]

#definition("Geometric Error")[
  The geometric error is the perpendicular distance from the point to the line:
  $ r_i^"geom"(bold(theta)) = (abs(a x_i + b y_i + c))/(sqrt(a^2 + b^2)) $
]

The relationship between algebraic and geometric errors is:
$
  r_i^"geom"(bold(theta)) = (abs(r_i^"alg"(bold(theta))))/(sqrt(a^2 + b^2))
$

=== Constrained Least Squares Formulation
The DLT problem is formulated as:
$ bold(theta)^* = argmin_(bold(theta)) norm(bold(A) bold(theta))_2^2 quad "subject to" quad norm(bold(theta))_2 = 1 $

where the design matrix is:
$
  bold(A) = mat(
    x_1, y_1, 1;
    x_2, y_2, 1;
    dots.v, dots.v, dots.v;
    x_n, y_n, 1
  ) in RR^(n times 3)
$

#theorem("DLT Solution via SVD")[
  Let $bold(A) = bold(U) bold(Sigma) bold(V)^T$ be the singular value decomposition of $bold(A)$, where:
  - $bold(U) in RR^(n times n)$ and $bold(V) in RR^(3 times 3)$ are orthogonal matrices
  - $bold(Sigma) = diag(sigma_1, sigma_2, sigma_3)$ with $sigma_1 >= sigma_2 >= sigma_3 >= 0$

  Then the optimal solution is $bold(theta)^* = bold(v)_3$, the last column of $bold(V)$.
] <dlt>

*Proof*:
Using the orthogonality of $bold(U)$ and the substitution $bold(w) = bold(V)^T bold(theta)$:
$
  norm(bold(A) bold(theta))_2^2 & = norm(bold(U) bold(Sigma) bold(V)^T bold(theta))_2^2                        \
                                & = norm(bold(Sigma) bold(V)^T bold(theta))_2^2 quad "(orthogonal invariance)" \
                                & = norm(bold(Sigma) bold(w))_2^2 = sum_(i=1)^3 sigma_i^2 w_i^2
$
Since $norm(bold(theta))_2 = norm(bold(w))_2 = 1$, minimizing the objective requires placing all weight on the smallest singular value:
$
  bold(w)^* = [0, 0, 1]^T => bold(theta)^* = bold(V) bold(w)^* = bold(v)_3
$
Numerical conditioning is crucial for stable DLT computation, especially when dealing with higher-order geometric entities.

#algorithm-figure("Direct Linear Transformation (DLT)", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([DLT], [data points ${(x_i, y_i)}_(i=1)^n$], {
    Comment[*Input:* Data points ${(x_i, y_i)}_(i=1)^n$]
    Comment[*Output:* Line parameters $bold(theta)^* = [a, b, c]^T$]
    LineBreak
    Comment[Construct design matrix:]
    Assign[$bold(A)$][$mat(
        x_1, y_1, 1;
        x_2, y_2, 1;
        dots.v, dots.v, dots.v;
        x_n, y_n, 1
      ) in RR^(n times 3)$]
    LineBreak
    Comment[Compute SVD of design matrix:]
    Assign[$bold(U), bold(Sigma), bold(V)$][$"SVD"(bold(A))$]
    Comment[where $bold(A) = bold(U) bold(Sigma) bold(V)^T$]
    LineBreak
    Comment[Extract solution from smallest singular vector:]
    Assign[$bold(theta)^*$][$bold(v)_3$]
    Comment[last column of $bold(V)$ corresponding to smallest singular value]
    LineBreak
    Comment[Optional: Normalize solution]
    If("normalization desired", {
      Assign[$bold(theta)^*$][$bold(theta)^* / norm(bold(theta)^*)_2$]
    })
    LineBreak
    Return[$bold(theta)^*$]
  })
})

#pagebreak()

=== Preconditioning for Numerical Stability
Numerical conditioning is crucial for stable DLT computation, especially when dealing with higher-order geometric entities.

#algorithm-figure("Normalized DLT", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([Normalized-DLT], [data points ${(x_i, y_i)}_(i=1)^n$], {
    Comment[*Input:* Data points ${(x_i, y_i)}_(i=1)^n$]
    LineBreak
    Comment[*Output:* Line parameters $bold(theta)^* = [a, b, c]^T$]
    LineBreak
    Comment[Compute data centroid and scale:]
    Assign[$bar(x)$][$1/n sum_(i=1)^n x_i$]
    LineBreak
    Assign[$bar(y)$][$1/n sum_(i=1)^n y_i$]
    LineBreak
    Comment[Apply normalization transformation:]
    LineBreak
    Assign[$bold(T)$][$mat(s_x, 0, -s_x bar(x); 0, s_y, -s_y bar(y); 0, 0, 1)$]
    Comment[where $s_x$ and $s_y$ are chosen such that the normalized data has unit variance]
    LineBreak
    Comment[Solve DLT for normalized points:]
    For($i = 1, dots, n$, {
      LineBreak
      Assign[$tilde(bold(p))_i$][$bold(T) mat(x_i; y_i; 1)$]
    })
    Assign[$tilde(bold(theta))^*$][$"DLT"({tilde(bold(p))_i})$]
    LineBreak
    Comment[Denormalize the solution:]
    Assign[$bold(theta)^*$][$bold(T)^T tilde(bold(theta))^*$]
    LineBreak
    Return[$bold(theta)^*$]
  })
})

#pagebreak()

== Robust Estimation Methods
#definition("Breakdown Point")[
  The breakdown point $epsilon^*$ of an estimator is the largest fraction of arbitrarily corrupted observations that the estimator can handle while still producing a bounded estimate close to the true value.
]

For ordinary least squares, the breakdown point is $epsilon^* = 0$, meaning a single outlier with sufficient leverage can arbitrarily corrupt the estimate. This motivates the development of robust alternatives.

M-estimators generalize maximum likelihood estimation by replacing the squared loss with robust alternatives:
$
  bold(theta)^* = argmin_(bold(theta)) sum_(i=1)^n rho(r_i (bold(theta)))
$

Common choices for $rho$ include:

#figure(
  table(
    columns: 3,
    align: left,
    table.header([*Name*], [*$rho(r)$*], [*Influence Function $psi(r) = rho'(r)$*]),
    [L2 (Least Squares)], [$r^2$], [$2r$],
    [L1 (Least Absolute)], [$abs(r)$], [$sign(r)$],
    [Huber],
    [$cases(r^2\/2 & abs(r) <= k, k abs(r) - k^2\/2 & abs(r) > k)$],
    [$cases(r & abs(r) <= k, k sign(r) & abs(r) > k)$],

    [Tukey's Bisquare],
    [$cases(k^2 slash 6[1-(1-r^2\/k^2)^3] & abs(r) <= k, k^2\/6 & abs(r) > k)$],
    [$cases(r(1-r^2\/k^2)^2 & abs(r) <= k, 0 & abs(r) > k)$],
  ),
  caption: [Common M-estimator loss functions and their derivatives],
)

#pagebreak()

== RANSAC: Random Sample Consensus
RANSAC takes a fundamentally different approach by explicitly modeling the presence of outliers through a consensus framework.

=== The Consensus Set
#definition("Consensus Set")[
  Given parameters $bold(theta)$ and threshold $epsilon$, the consensus set is:
  $
    cal(C)(bold(theta)) = {i : abs(r_i (bold(theta))) <= epsilon}
  $
]

RANSAC maximizes the cardinality of the consensus set:
$
  bold(theta)^* = argmax_(bold(theta)) abs(cal(C)(bold(theta)))
$

=== The RANSAC Algorithm
#algorithm-figure("RANSAC", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([RANSAC], ([Data points $cal(D)$], [inlier threshold $epsilon$], [confidence $p$]), {
    Comment[*Input:* Data points $cal(D)$, inlier threshold $epsilon$, confidence $p$]
    Comment[*Output:* Model parameters $bold(theta)^*$]
    LineBreak
    Comment[Initialize:]
    Assign[$cal(C)^*$][$emptyset$]
    Assign[$k$][$0$]
    LineBreak
    While($k < N$, {
      Comment[(a) Randomly sample minimal set $cal(S) subset cal(D)$ with $abs(cal(S)) = m$]
      Assign[$cal(S)$][randomly sample $m$ points from $cal(D)$]
      LineBreak
      Comment[(b) Fit model:]
      Assign[$bold(theta)_k$][$"FitModel"(cal(S))$]
      LineBreak
      Comment[(c) Evaluate consensus:]
      Assign[$cal(C)_k$][${i : abs(r_i (bold(theta)_k)) <= epsilon}$]
      LineBreak
      Comment[(d) if $abs(cal(C)_k) > abs(cal(C)^*)$ then update best model]
      If($abs(cal(C)_k) > abs(cal(C)^*)$, {
        Assign[$cal(C)^*$][$cal(C)_k$]
        Assign[$bold(theta)^*$][$bold(theta)_k$]
      })
      LineBreak
      Assign[$k$][$k + 1$]
    })
    LineBreak
    Comment[Refine:]
    Assign[$bold(theta)^*$][$"LeastSquares"(cal(C)^*)$]
    LineBreak
    Return[$bold(theta)^*$]
  })
})


=== Determining the Number of Iterations
The number of iterations $N$ is determined probabilistically to ensure finding at least one outlier-free sample with confidence $p$:

#theorem("RANSAC Iteration Count")[
  Given inlier ratio $w$ and desired confidence $p$, the required number of iterations is:
  $ N = (log(1-p))/(log(1-w^m)) $
  where $m$ is the minimal sample size.
]

*Proof:*
The probability of selecting an all-inlier minimal sample is $w^m$. The probability of failing to select such a sample in $N$ attempts is $(1-w^m)^N$. Setting this equal to $1-p$ and solving for $N$ yields the result.

#example("RANSAC Iteration Example")[
  For line fitting ($m=2$) with 50% inliers ($w=0.5$) and 99% confidence ($p=0.99$):
  $ N = (log(0.01))/(log(1-0.5^2)) = (log(0.01))/(log(0.75)) approx 17 $
]

== MSAC and MLESAC Variants
=== MSAC: M-estimator Sample Consensus
MSAC refines RANSAC by considering the magnitude of residuals within the inlier band:

$ L_"MSAC"(bold(theta)) = sum_(i=1)^n min(abs(r_i (bold(theta))), epsilon) $

This formulation corresponds to a truncated L1 loss, providing better discrimination between competing models with similar consensus set sizes.

=== MLESAC: Maximum Likelihood Sample Consensus
MLESAC models the error distribution explicitly as a mixture:
$ p(r_i) = gamma cal(N)(0, sigma^2) + (1-gamma) cal(U)(-v, v) $

where $gamma$ is the inlier ratio, and maximizes the likelihood of the observed residuals.

#pagebreak()
