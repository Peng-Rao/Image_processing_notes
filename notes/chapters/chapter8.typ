#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *
#import "@preview/algorithmic:1.0.2"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

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

  *Proof:*
  The Frobenius norm can be expressed as:
  $
    norm(bold(A) - bold(u) bold(v)^T)_F^2 &= norm(bold(A))_F^2 - 2 "trace"(bold(A)^T bold(u) bold(v)^T) + norm(bold(u) bold(v)^T)_F^2 \
    &= norm(bold(A))_F^2 - 2 bold(v)^T bold(A)^T bold(u) + norm(bold(v))_2^2
  $
] <thm:rank_one>

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

#algorithm-figure(
  "K-SVD Dictionary Learning Algorithm",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      [K-SVD],
      (
        [$bold(Y)$],
        [$m$],
        [$T_0$],
        [$K$],
      ),
      {
        Comment[*Input:* Training signals $bold(Y) in RR^(n times N)$, dictionary size $m$, sparsity level $T_0$, max iterations $K$]
        Comment[*Output:* Dictionary $bold(D) in RR^(n times m)$ and sparse coefficients $bold(X) in RR^(m times N)$]
        LineBreak
        Comment[Initialize dictionary:]
        Assign[$bold(D)^((0))$][random matrix with unit-norm columns]
        Assign[$k$][$0$]
        LineBreak
        Comment[Main K-SVD loop:]
        While(
          $k < K and "not converged"$,
          {
            Comment[SPARSE CODING PHASE:]
            Comment[For each training signal $bold(y)_i$, solve sparse coding problem]
            For(
              $i = 1, dots, N$,
              {
                Assign[$bold(x)_i^((k+1))$][$arg min_(bold(x)) norm(bold(y)_i - bold(D)^((k)) bold(x))_2^2$ subject to $norm(bold(x))_0 <= T_0$]
                Comment[Use OMP, MP, or other sparse coding algorithm]
              },
            )
            LineBreak
            Comment[DICTIONARY UPDATE PHASE:]
            Comment[Update each dictionary column sequentially]
            For($j = 1, dots, m$, {
              Comment[(a) Compute error matrix excluding $j$-th atom:]
              Assign[$bold(E)_j$][$bold(Y) - sum_(ell != j) bold(d)_ell bold(x)_ell^T$]
              LineBreak
              Comment[(b) Identify support set:]
              Assign[$Omega_j$][${i : x_(j,i) != 0}$]
              LineBreak
              Comment[(c) Check if atom is used:]
              If($Omega_j = emptyset$, {
                Comment[Replace unused atom with random unit vector]
                Assign[$bold(d)_j$][random unit vector]
              })
              Else({
                Comment[(d) Extract restricted error matrix:]
                Assign[$bold(E)_j^R$][$bold(E)_j (:, Omega_j)$]
                LineBreak
                Comment[(e) Compute SVD of restricted matrix:]
                Assign[$bold(U), bold(Sigma), bold(V)$][$"SVD"(bold(E)_j^R)$]
                LineBreak
                Comment[(f) Update dictionary column:]
                Assign[$bold(d)_j$][$bold(u)_1$]
                Comment[First left singular vector]
                LineBreak
                Comment[(g) Update sparse coefficients:]
                Assign[$bold(x)_j^R$][$sigma_1 bold(v)_1$]
                Comment[Scaled first right singular vector]
                LineBreak
                Comment[(h) Restore to full representation:]
                Assign[$bold(x)_j^T (Omega_j)$][$bold(x)_j^R$]
                Assign[$bold(x)_j^T ("complement"(Omega_j))$][$bold(0)$]
              })
            })
            LineBreak
            Comment[Check convergence:]
            If($norm(bold(Y) - bold(D)^((k+1)) bold(X)^((k+1)))_F < epsilon$, {
              Comment[Convergence achieved]
              Break
            })
            LineBreak
            Assign[$k$][$k + 1$]
          },
        )
        LineBreak
        Return[$bold(D)^((k)), bold(X)^((k))$]
      },
    )
  },
)

#pagebreak()

== Anomaly Detection
Let $cal(I) subset bb(R)^(H times W)$ represent the space of grayscale images with height $H$ and width $W$. Given an input image $I in cal(I)$ containing fiber material, our objective is to construct a binary anomaly mask $M in {0,1}^(H times W)$ such that:
$
  M(i,j) = cases(
    1 & "if pixel " (i,j) " belongs to an anomalous region",
    0 & "if pixel " (i,j) " belongs to a normal region"
  )
$ <mask_definition>

#definition("Anomaly Detection Problem")[
  Given a training set $cal(T) = {I_1, I_2, ..., I_N}$ of normal (defect-free) images and a test image $I_"test"$, the anomaly detection problem consists of learning a decision function $f: cal(I) -> [0,1]^(H times W)$ that assigns an anomaly score to each pixel, where higher scores indicate higher probability of anomaly.
] <anomaly_detection>

The fundamental challenge lies in the fact that we possess only examples of normal fiber structures during training, making this an unsupervised or one-class classification problem.

With a learned dictionary representing normal patterns, we can now formulate the anomaly detection algorithm. The approach proceeds by analyzing each patch in the test image and computing anomaly scores based on reconstruction quality and sparsity characteristics.

=== Anomaly Score Computation
For a test patch $P_"test"$, we compute its sparse representation:
$
  hat(alpha)_"test" = argmin_alpha 1/2 ||P_"test" - bold(D) alpha||_2^2 + lambda ||alpha||_1
$ <test_sparse_coding>
Several anomaly scores can be derived from this representation:

=== Reconstruction Error Score <reconstruction_score>
The reconstruction error quantifies how well the learned dictionary can represent the test patch:
$
  S_"rec"(P_"test") = ||P_"test" - bold(D) hat(alpha)_"test"||_2^2
$ <reconstruction_error>

#lemma("Reconstruction Error for Normal Patches")[
  If $P_"test"$ belongs to the same distribution as the training patches and the dictionary is sufficiently expressive, then $S_"rec"(P_"test") <= epsilon$ for small $epsilon > 0$.
] <normal_reconstruction>

=== Sparsity Score <sparsity_score>

The sparsity score measures the number of dictionary atoms required for representation:
$
  S_"sparse"(P_"test") = ||hat(alpha)_"test"||_0
$ <sparsity_score>

Alternative sparsity measures include the $ell_1$ norm and Gini coefficient:
$
  S_"sparse"^((1))(P_"test") & = ||hat(alpha)_"test"||_1
$
<l1_sparsity>
$
  S_"sparse"^"Gini"(P_"test") & = (2 sum_(i=1)^K i dot alpha_((i)))/(K sum_(i=1)^K alpha_((i))) - (K+1)/K
$ <gini_sparsity>

where $alpha_((i))$ denotes the $i$-th largest coefficient in magnitude.

=== Combined Anomaly Score <combined_score>
A robust anomaly score combines reconstruction error and sparsity information:
$
  S(P_"test") = beta S_"rec"(P_"test") + (1-beta) S_"sparse"(P_"test")
$ <combined_score_eq>
where $beta in [0,1]$ balances the two components.

For numerical stability and interpretability, scores are often normalized using statistics from the training set:
$
  S_"norm"(P_"test") = (S(P_"test") - mu_S)/sigma_S
$ <normalized_score>
where $mu_S$ and $sigma_S$ are the mean and standard deviation of scores computed on training patches.

=== Pixel-Level Anomaly Mapping <pixel_mapping>
Since patches overlap, multiple anomaly scores are computed for each pixel. We aggregate these scores using various strategies:

+ *Maximum Aggregation*: $S_"pixel"(i,j) = max_(P in.rev (i,j)) S(P)$
+ *Average Aggregation*: $S_"pixel"(i,j) = 1/(|cal(P)_(i,j)|) sum_(P in cal(P)_(i,j)) S(P)$
+ *Weighted Average*: $S_"pixel"(i,j) = sum_(P in cal(P)_(i,j)) w_P S(P)$

where $cal(P)_(i,j)$ denotes the set of patches containing pixel $(i,j)$.

For weighted averaging, a common choice is Gaussian weighting based on distance from patch center:
$
  w_P (i,j) = exp(-(||(i,j) - "center"(P)||_2^2)/(2 sigma_w^2))
$ <gaussian_weights>

=== Threshold Selection and Binary Mask Generation
The final step converts continuous anomaly scores to binary decisions. Several threshold selection strategies are employed:

==== Percentile-Based Thresholding
Set the threshold to the $p$-th percentile of training scores:
$
  tau_p = "percentile"(p, {S(P_i)}_(i=1)^m)
$ <percentile_threshold_eq>
Typical values include $p = 95%$ or $p = 99%$.

==== Statistical Thresholding <statistical_threshold>
Assume training scores follow a known distribution (e.g., Gaussian) and set:
$
  tau_"stat" = mu_S + k sigma_S
$ <statistical_threshold_eq>
where $k$ controls the false positive rate (e.g., $k = 2$ for approximately 2.5% false positives).

==== ROC-Based Thresholding
When validation data with known anomalies is available, select the threshold maximizing the Youden index:
$
  tau_"Youden" = argmax_tau ("Sensitivity"(tau) + "Specificity"(tau) - 1)
$ <youden_threshold>

The binary anomaly mask is then generated as:
$
  M(i,j) = cases(
    1 & "if " S_"pixel"(i,j) > tau,
    0 & "otherwise"
  )
$ <final_mask>

#pagebreak()
