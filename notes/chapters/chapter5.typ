#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *

= Sparse Coding
== The Sparse Coding Problem
Given the desire for sparse representations, a natural formulation is to seek the sparsest possible $bold(alpha)$ such that:
$
  bold(x) = D bold(alpha)
$
This leads to the following optimization problem:
$
  min_(bold(alpha) in RR^n) norm(bold(alpha))_0 quad "subject to" quad bold(x) = D bold(alpha)
$ <eq:p0_problem>
This is often referred to as the *$P_0$ problem*.

#align(center)[
  #box(fill: gray.lighten(90%), stroke: 1pt + black, inset: 10pt, width: 85%, [
    *Goal:* Among all solutions $bold(alpha)$ such that $D bold(alpha) = bold(x)$, select the sparsest one.
  ])
]

*Challenge:* The $ell_0$ norm is non-convex, discontinuous, and leads to combinatorial complexity. Solving @eq:p0_problem exactly is NP-hard in general.

=== Union of Subspaces Interpretation
Let us assume $D in RR^(m times n)$ is a dictionary with $n > m$, i.e., an overcomplete dictionary.

Suppose we restrict $bold(alpha)$ to have at most $s$ non-zero entries. Then the image $D bold(alpha)$ lies in a subspace spanned by $s$ columns of $D$.

#definition("Sparsity-Induced Subspace")[
  If $bold(alpha)$ has $norm(bold(alpha))_0 <= s$, then $bold(x) = D bold(alpha)$ lies in a subspace $S_omega := "span"(D_omega)$, where $omega$ is the support of $bold(alpha)$ and $D_omega$ denotes the sub-matrix of $D$ restricted to columns indexed by $omega$.
]

Therefore, the space of all $s$-sparse representations is the union of all such $s$-dimensional subspaces:
$
  cal(M)_s := union.big_(omega subset {1, dots, n}, |omega| <= s) "span"(D_omega)
$

#align(center)[
  #box(
    fill: gray.lighten(90%),
    stroke: 1pt + black,
    inset: 10pt,
    width: 85%,
    [
      *Interpretation:* Sparse modeling corresponds to finding the best subspace (among exponentially many) in which to approximate the signal $bold(x)$.
    ],
  )
]

*Key Point:* Unlike PCA (which projects onto a single global subspace), sparse coding projects onto a _union of low-dimensional subspaces_, selected adaptively based on the input $bold(x)$.

=== A 2D Illustration of Sparsity

Suppose we are working in $RR^2$ and $D$ has 3 atoms:
$
  D = mat(bold(d)_1, bold(d)_2, bold(d)_3), quad D in RR^(2 times 3)
$
Each $bold(d)_i$ is a column vector in $RR^2$.

Let $bold(x) in RR^2$ be a signal we wish to approximate. If we restrict $norm(bold(alpha))_0 = 1$, then the approximant $D bold(alpha)$ must lie along one of the directions $bold(d)_1$, $bold(d)_2$, or $bold(d)_3$.

#figure(
  canvas({
    import draw: *

    set-style(
      stroke: (thickness: 1pt, cap: "round"),
      mark: (fill: black, scale: 1.2),
    )

    // Draw the three dictionary atoms
    line((0, 0), (3, 0), mark: (end: "stealth"))
    content((3.2, 0), [$bold(d)_1$], anchor: "west")

    line((0, 0), (2.1, 2.1), mark: (end: "stealth"))
    content((2.3, 2.3), [$bold(d)_2$], anchor: "south-west")

    line((0, 0), (0, 3), mark: (end: "stealth"))
    content((0, 3.2), [$bold(d)_3$], anchor: "south")

    // Draw the signal vector
    set-style(stroke: (paint: blue, thickness: 2pt))
    line((0, 0), (1.8, 1.35), mark: (end: "stealth"))
    content((2, 1.5), text(fill: blue, [$bold(x)$]), anchor: "west")
  }),
  caption: [Approximation of $bold(x)$ via projections onto sparse atoms],
)

*Interpretation:*
- We seek the atom $bold(d)_i$ such that the projection of $bold(x)$ onto $"span"(bold(d)_i)$ minimizes the residual.
- This is the best 1-sparse approximation.

*In general:*
If $norm(bold(alpha))_0 <= s$, the approximation lives in a union of $binom(n, s)$ subspaces.

=== Combinatorial Intractability
*Why is $P_0$ hard?* To find the optimal $s$-sparse representation of $bold(x)$, one must:
+ Enumerate all subsets $omega subset {1, dots, n}$ of size $s$
+ Solve the least squares problem:
  $
    bold(alpha)_omega = arg min_(bold(z) in RR^s) norm(D_omega bold(z) - bold(x))_2^2
  $
+ Select the best $omega$ minimizing the residual.

The number of subsets grows exponentially:
$
  "#subspaces" = binom(n, s) tilde (n e / s)^s
$

*Illustrative Example:*
Let $n = 1000$, $s = 20$. Then:
$
  binom(1000, 20) approx 10^34
$
Assuming a billion operations per second, exhaustive search would take more time than the age of the universe.

#align(center)[
  #box(
    fill: gray.lighten(90%),
    stroke: 1pt + black,
    inset: 10pt,
    width: 85%,
    [
      *Conclusion:* The $ell_0$ sparse coding problem is combinatorially explosive. Efficient approximations are necessary.
    ],
  )
]

#pagebreak()

== Greedy Algorithms for Sparse Coding
Given the computational intractability of exact $ell_0$ minimization, we turn to greedy approximation algorithms that provide computationally feasible solutions.

A *greedy algorithm* for sparse coding makes locally optimal choices at each iteration without reconsidering previous decisions, building up the solution incrementally by adding one dictionary atom at a time.

#example("Coin Change Analogy")[
  The greedy approach mirrors the coin change problem:
  - *Goal*: Minimize the number of coins to make change
  - *Greedy strategy*: Always use the largest denomination possible
  - *Limitation*: Optimal only for specially designed coin systems

  For standard currency systems (e.g., {1, 2, 5, 10, 20, 50}), greedy gives optimal solutions. However, for pathological systems (e.g., {1, 3, 4}), greedy fails: making change for 6 units gives greedy solution 4+1+1 (3 coins) vs. optimal 3+3 (2 coins).
] <ex:coin_change>

#pagebreak()

== Matching Pursuit Algorithm
The *Matching Pursuit (MP)* algorithm embodies the greedy principle for sparse coding:
*Input:* Signal $bold(y)$, dictionary $bold(D)$ (normalized), stopping criterion \
*Output:* Sparse representation $bold(x)$

+ *Initialize:*
  $
    bold(x)^((0)) & = bold(0) quad      & "(coefficient vector)" \
    bold(r)^((0)) & = bold(y) quad      &           "(residual)" \
      Omega^((0)) & = emptyset.rev quad &         "(active set)" \
                k & = 0 quad            &  "(iteration counter)"
  $

+ *Sweep Stage:* For each atom $j = 1, dots, n$, compute the approximation error:
  $
    E_j^((k)) = norm(bold(r)^((k)) - angle.l bold(d)_j \, bold(r)^(k) angle.r bold(d)_j)_2^2
  $ <eq:mp_error>

+ *Atom Selection:* Choose the atom with minimum error:
  $
    j^* = arg min_(j=1,dots,n) E_j^((k))
  $ <eq:mp_selection>

  Equivalently (by maximizing correlation):
  $
    j^* = arg max_(j=1,dots,n) angle.l bold(d)_j \, bold(r)^(k) angle.r
  $ <eq:mp_correlation>

+ *Coefficient Update:* Compute the projection coefficient:
  $
    z_(j^*)^((k)) = angle.l bold(d)_(j^*) \, bold(r)^(k) angle.r
  $ <eq:mp_coefficient>

+ *Solution Update:*
  $
    bold(x)^((k+1)) = bold(x)^((k)) + z_(j^*)^((k)) bold(e)_(j^*)
  $ <eq:mp_solution_update>
  where $bold(e)_(j^*)$ is the $j^*$-th standard basis vector.

+ *Residual Update:*
  $
    bold(r)^((k+1)) = bold(r)^((k)) - z_(j^*)^((k)) bold(d)_(j^*)
  $ <eq:mp_residual_update>

+ *Active Set Update:*
  $
    Omega^((k+1)) = Omega^((k)) union {j^*}
  $

+ *Stopping Criteria:* Terminate if:
  - $|Omega^((k+1))| >= k_"max"$ (maximum sparsity reached)
  - $norm(bold(r)^((k+1)))_2 <= epsilon$ (residual threshold met)

  Otherwise, set $k arrow.l k+1$ and return to step 2.

#pagebreak()

=== Residual Monotonicity
The Matching Pursuit algorithm produces a monotonically decreasing sequence of residual norms:
$
  norm(bold(r)^((k+1)))_2 <= norm(bold(r)^((k)))_2
$
with strict inequality unless $bold(r)^((k))$ is orthogonal to all dictionary atoms.

=== Atom Reselection
Unlike orthogonal methods, Matching Pursuit may select the same atom multiple times in successive iterations. This occurs because:
+ The algorithm does not enforce orthogonality of residuals to previously selected atoms
+ Residual components may align with previously selected atoms after updates
+ This can lead to slower convergence compared to orthogonal variants

=== Convergence Analysis
For any finite dictionary $bold(D)$ and signal $bold(y)$, the Matching Pursuit algorithm converges in the sense that:
$
  lim_(k -> oo) norm(bold(r)^((k)))_2 = min_(bold(x)) norm(bold(y) - bold(D) bold(x))_2
$
Furthermore, if $bold(y) in "span"(bold(D))$, then the algorithm achieves exact recovery in finite steps.

=== Approximation Quality
While Matching Pursuit provides computational tractability, it may not achieve the globally optimal sparse solution. The quality of approximation depends on the coherence structure of the dictionary.

The *coherence* of a dictionary $bold(D)$ with normalized columns is:
$
  mu(bold(D)) = max_(i != j) |bold(d)_i^T bold(d)_j|
$
Under certain conditions on dictionary coherence and signal sparsity, Matching Pursuit provides approximation guarantees. Specifically, if the true sparse representation has sparsity $k$ and the dictionary satisfies appropriate coherence conditions, then MP recovers a solution with controlled approximation error.

=== Computational Complexity
Each iteration of Matching Pursuit requires:
- $cal(O)(m n)$ operations for the sweep stage (computing all correlations)
- $cal(O)(m)$ operations for residual update
- Total per-iteration complexity: $cal(O)(m n)$

For $k$ iterations, the total complexity is $cal(O)(k m n)$, which is polynomial and practically feasible.

#pagebreak()

== Orthogonal Matching Pursuit
*Orthogonal Matching Pursuit (OMP)* is a variant of Matching Pursuit that enforces orthogonality of the residuals to previously selected atoms. This leads to improved convergence properties and better approximation quality.

The OMP algorithm embodies the greedy principle for sparse coding with orthogonal projections:

*Input:* Signal $bold(y)$, dictionary $bold(D)$ (normalized), stopping criterion \
*Output:* Sparse representation $bold(x)$

+ *Initialize:*
  $
    bold(x)^((0)) & = bold(0) quad      & "(coefficient vector)" \
    bold(r)^((0)) & = bold(y) quad      &           "(residual)" \
      Omega^((0)) & = emptyset.rev quad &         "(active set)" \
                k & = 0 quad            &  "(iteration counter)"
  $

+ *Atom Selection:* Choose the atom with maximum correlation:
  $
    j^* = arg max_(j=1,dots,n) |angle.l bold(d)_j \, bold(r)^((k)) angle.r|
  $ <eq:omp_selection>

+ *Active Set Update:*
  $
    Omega^((k+1)) = Omega^((k)) union {j^*}
  $

+ *Orthogonal Projection:* Solve the least squares problem over the active set:
  $
    bold(alpha)_Omega^((k+1)) = arg min_(bold(z)) norm(bold(D)_Omega bold(z) - bold(y))_2^2
  $ <eq:omp_projection>

  where $bold(D)_Omega$ is the sub-matrix of $bold(D)$ with columns indexed by $Omega^((k+1))$.

+ *Solution Update:*
  $
    bold(x)^((k+1))_j = cases(
      alpha_j^((k+1)) quad & "if" j in Omega^((k+1)),
      0 quad & "otherwise"
    )
  $ <eq:omp_solution_update>

+ *Residual Update:* Compute the orthogonal residual:
  $
    bold(r)^((k+1)) = bold(y) - bold(D)_Omega bold(alpha)_Omega^((k+1))
  $ <eq:omp_residual_update>

+ *Stopping Criteria:* Terminate if:
  - $|Omega^((k+1))| >= k_"max"$ (maximum sparsity reached)
  - $norm(bold(r)^((k+1)))_2 <= epsilon$ (residual threshold met)

  Otherwise, set $k arrow.l k+1$ and return to step 2.

#pagebreak()

== OMP-Based Image Denoising
The integration of OMP into image denoising frameworks requires careful consideration of patch processing, dictionary design, and aggregation strategies.

Natural images exhibit strong local correlations but varying global statistics. The patch-based approach decomposes the image into overlapping patches, each processed independently:

*Input:* Noisy image $bold(Y) in RR^(N times M)$, dictionary $bold(D) in RR^(n times m)$, sparsity level $s$ \
*Output:* Denoised image $hat(bold(Y))$

+ *Patch Extraction:* For image $bold(Y) in RR^(N times M)$, extract patches $(bold(y)_i)_(i=1)^P$ where each $bold(y)_i in RR^n$ represents a vectorized $sqrt(n) times sqrt(n)$ patch.

+ *Mean Computation and Centering:* For each patch $bold(y)_i$:
  $
                mu_i & = 1/n sum_(j=1)^n y_(i,j)  \
    tilde(bold(y))_i & = bold(y)_i - mu_i bold(1)
  $
  where $bold(1) in RR^n$ is the vector of all ones.

+ *Sparse Coding:* Apply OMP to each mean-centered patch:
  $
    hat(bold(alpha))_i = "OMP"(bold(D), tilde(bold(y))_i, s)
  $

+ *Reconstruction:* Compute denoised patches by adding back the mean:
  $
    hat(bold(x))_i = bold(D) hat(bold(alpha))_i + mu_i bold(1)
  $

+ *Aggregation:* Reconstruct the full image by averaging overlapping reconstructions at each pixel location.

== Linearity Analysis of Sparse Coding Algorithms

A fundamental question in sparse coding concerns the linearity properties of the resulting algorithms. An algorithm $cal(A)$ is considered linear if and only if it satisfies:

#definition("Linear Algorithm")[
  An algorithm $cal(A): RR^m -> RR^n$ is linear if and only if for all $alpha, beta in RR$ and $bold(y)_1, bold(y)_2 in RR^m$:
  $
    cal(A)(alpha bold(y)_1 + beta bold(y)_2) = alpha cal(A)(bold(y)_1) + beta cal(A)(bold(y)_2)
  $
] <def:linear_algorithm>

#pagebreak()

== Uniqueness Guarantees for Sparse Solutions
The following theorem provides conditions under which the solution to the $ell_0$ constraint problem is unique.

#theorem([Uniqueness of $ell_0$ Solutions])[
  Consider the system $bold(D) bold(x) = bold(y)$ where $bold(D) in RR^{m times n}$. If there exists a solution $hat(bold(x))$ such that:
  $
    norm(hat(bold(x)))_0 < 1/2 "spark"(bold(D))
  $
  then $hat(bold(x))$ is the unique solution to the $ell_0$ constraint problem.

  *Proof:*
  Suppose, for the sake of contradiction, that there exists another solution $tilde(bold(x)) != hat(bold(x))$ such that $bold(D) tilde(bold(x)) = bold(y)$.

  Since both $hat(bold(x))$ and $tilde(bold(x))$ satisfy the linear system:
  $
      bold(D) hat(bold(x)) & = bold(y) \
    bold(D) tilde(bold(x)) & = bold(y)
  $

  Subtracting these equations yields:
  $
    bold(D)(hat(bold(x)) - tilde(bold(x))) = bold(0)
  $

  This shows that $hat(bold(x)) - tilde(bold(x))$ is a non-zero solution to the homogeneous system. By @lem:spark_homogeneous:
  $
    "spark"(bold(D)) <= norm(hat(bold(x)) - tilde(bold(x)))_0
  $

  Using the triangle inequality for the $ell_0$ pseudo-norm:
  $
    norm(hat(bold(x)) - tilde(bold(x)))_0 <= norm(hat(bold(x)))_0 + norm(tilde(bold(x)))_0
  $

  Combining these inequalities:
  $
    "spark"(bold(D)) <= norm(hat(bold(x)))_0 + norm(tilde(bold(x)))_0
  $

  Since $tilde(bold(x))$ is also assumed to be a solution to the $ell_0$ problem, and $hat(bold(x))$ is the optimal solution:
  $
    norm(tilde(bold(x)))_0 >= norm(hat(bold(x)))_0
  $

  Therefore:
  $
    "spark"(bold(D)) <= 2 norm(hat(bold(x)))_0
  $

  This contradicts our assumption that $norm(hat(bold(x)))_0 < 1/2 "spark"(bold(D))$. Hence, $hat(bold(x))$ is unique.
] <thm:l0_uniqueness>

#attention("Practical Limitations")[
  While theoretically elegant, the uniqueness conditions are often too restrictive in practice:
  - Computing $"spark"(bold(D))$ is computationally intractable for large matrices
  - The bound $norm(hat(bold(x)))_0 < 1/2 "spark"(bold(D))$ is often very conservative
  - Real-world signals may not satisfy the sparsity requirements
]

#pagebreak()

== Application to Image Inpainting
Image inpainting addresses the reconstruction of missing or corrupted pixels in digital images. Using sparse coding theory, we can formulate inpainting as a sparse reconstruction problem.

Consider an image patch $bold(s)_0 in RR^n$ (vectorized) and its corrupted version $bold(s) in RR^n$ where some pixels are missing or corrupted. The relationship between them can be expressed as:
$
  bold(s) = bold(Omega) bold(s)_0
$
where $bold(Omega) in RR^{n times n}$ is a diagonal matrix with:
$
  Omega_(i i) = cases(
    1 quad & "if pixel " i " is known",
    0 quad & "if pixel " i " is missing"
  )
$

Assuming the original patch admits a sparse representation:
$
  bold(s)_0 = bold(D) bold(x)_0
$

where $bold(D) in RR^(n times m)$ is an overcomplete dictionary and $bold(x)_0$ is sparse, the corrupted patch becomes:
$
  bold(s) = bold(Omega) bold(D) bold(x)_0 = bold(D)_Omega bold(x)_0
$
where $bold(D)_Omega = bold(Omega) bold(D)$ represents the "inpainted dictionary."

The key insight is that the spark of the inpainted dictionary relates to the original dictionary:
#proposition("Spark of Inpainted Dictionary")[
  For the inpainted dictionary $bold(D)_Omega = bold(Omega) bold(D)$:
  $
    "spark"(bold(D)_Omega) >= "spark"(bold(D))
  $
] <prop:inpainted_spark>

*Proof:*
Removing rows (zeroing out pixels) from a matrix cannot decrease the spark, as linear dependencies between columns are preserved or potentially eliminated.

Let $bold(s)_0$ be an image patch with sparse representation $bold(s)_0 = bold(D) bold(x)_0$ where $norm(bold(x)_0)_0 < 1/2 "spark"(bold(D)_Omega)$. Then:
+ The sparse coding problem $min_(bold(x)) norm(bold(x))_0$ subject to $bold(D)_Omega bold(x) = bold(s)$ has a unique solution $bold(x)_0$
+ The reconstruction $hat(bold(s))_0 = bold(D) bold(x)_0$ perfectly recovers the original patch

*Proof:*
The proof follows directly from @thm:l0_uniqueness applied to the inpainted dictionary $bold(D)_Omega$.

*Sparse Coding Inpainting Algorithm*

*Input:* Corrupted image patch $bold(s)$, dictionary $bold(D)$, mask $bold(Omega)$

+ *Construct inpainted dictionary:* $bold(D)_Omega = bold(Omega) bold(D)$
+ *Solve sparse coding:* $hat(bold(x)) = arg min_(bold(x)) norm(bold(x))_0$ subject to $bold(D)_Omega bold(x) = bold(s)$
+ *Reconstruct patch:* $hat(bold(s))_0 = bold(D) hat(bold(x))$

*Output:* Inpainted patch $hat(bold(s))_0$

In practice, step 2 is solved using OMP with the inpainted dictionary:
$
  hat(bold(x)) = "OMP"(bold(D)_Omega, bold(s), K)
$

where $K$ is a predetermined sparsity level. The key insight is that the synthesis step uses the original dictionary $bold(D)$, not the inpainted dictionary $bold(D)_Omega$.

#pagebreak()

