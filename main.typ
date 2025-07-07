#import "@local/simple-note:0.0.1": *
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot

#show: zebraw
#show: simple-note.with(
  title: [ Image Processing ],
  date: datetime(year: 2025, month: 2, day: 17),
  authors: (
    (
      name: "Rao",
      github: "https://github.com/Peng-Rao",
      homepage: "https://github.com/Peng-Rao",
    ),
  ),
  affiliations: (
    (
      id: "1",
      name: "Politecnico di Milano",
    ),
  ),
  // cover-image: "./figures/polimi_logo.png",
  background-color: "#DDEEDD",
)
#set math.mat(delim: "[")
#set math.vec(delim: "[")
#set math.equation(supplement: [Eq.])

#let nonum(eq) = math.equation(block: true, numbering: none, eq)
#let firebrick(body) = text(fill: rgb("#b22222"), body)

= Sparsity and Parsimony
The principle of sparsity or "parsimony" consists in representing some phenomenon with as few variable as possible. Stretch back to philosopher William Ockham in the 14th century, Wrinch and Jeffreys relate simplicity to parsimony:

The *existence of simple laws* is, then, apparently, to be regarded as *a quality of nature*; and accordingly we may infer that it is justifiable to *prefer a simple law to a more complex one that fits our observations slightly better*.

== Sparsity in Statistics
Sparsity is used to *prevent overfitting and improve interpretability of learned models*. In model fitting, the number of parameters is typically used as a criterion to perform model selection. See Bayes Information Criterion (BIC), Akaike Information Criterion (AIC), ...., Lasso.

== Sparsity in Signal Processing
*Signal Processing*: similar concepts but different terminology. *Vectors corresponds to signals* and data modeling is crucial for performing various operations such as *restoration, compression, solving inverse problems*.

Signals are approximated by sparse linear combinations of *prototypes*(basis elements / atoms of a dictionary), resulting in simpler and compact model.

#figure(
  image("figures/enforce-sparsity.png"),
  caption: "Enforce sparsity in signal processing",
)

#pagebreak()

= Signal Processing
== Discrete Cosine Transform (DCT)
=== 1D DCT
Generate the DCT basis according to the following formula, the $k$-th atom of the DCT basis in dimension $M$ is defined as:
$
  "DCT"_k(n) = c_k cos(k pi (2n + 1) / (2M)) space space space n, k = 0, 1, ..., M-1
$
where $c_0=sqrt(1 / M)$ and $c_k=sqrt(2 / M)$ for $k eq.not 0$.

For each $k=0, dots, M-1$, just sample each function
$
  "DCT"_k(n) = cos(k pi (2n + 1) / (2M))
$
at $n=0, dots, M-1$, obtain a vector. Ignore the normalization coefficient. Divide each vector by its $ell_2$ norm.

Mathematically, suppose the image signal is $s in RR^M$.
$
  x="dct2"(s)=D^T s
$
where $D^T$ represents the *DCT basis matrix*. $x$ contains the *DCT coefficients*, which are a *sparse representation* of $s$.

The *inverse DCT transformation* reconstructs $s$ from $x$:
$
  s="idct2"(x)=D x
$

=== 2D DCT

*2D Discrete Cosine Transform* (DCT) can be used as a dictionary for representing image patches. A small patch of an image is extracted, represented as $s$, with dimension $p times p$. This patch can be *flattened* into a vector of length $M=p^2$, meaning each patch is reshaped into a vector of length $M$. The *2D-DCT* is used to transform the patch $s$ into DCT coefficients $x$.

Suppose the image signal is $S in RR^(M times N)$.The 2D DCT can be decomposed into two *1D DCT* operations:
+ *Column-wise DCT*: apply *1D DCT* to each column of the image patch: $Zeta=D^T S$
+ *Row-wise DCT*: apply *1D DCT* to each column of the image patch: $X^T=D^T Zeta^T arrow X=D^T S D$

#example("JPEG Compression")[
  The image is divided into non-overlapping $8 times 8$ blocks. Each block is treated separately during the compression process.

  For each $8 times 8$ block, the *DCT* is applied, transforming pixel values into frequency-domain coefficients. Each $8 times 8$ block's coefficients are checked against a compression threshold $tau$, coefficients with absolute values below $tau$ are *discarded*(set to zero). The larger the threshold $tau$, the more coefficients are discarded, leading to *higher compression*.

  The compression ratio is defined as:
  $
    "Comp Ratio" = 1 - "#Non-zero coefficients" / "#Pixels in the image"
  $
  To measure how much the image quality is degraded after compression, *Peak Signal-to-Noise Ratio (PSNR)* is used:
  $
    "PSNR" = 10 log_10 (1 / "MSE"(Y, hat(Y)))
  $
]

#pagebreak()

= Image Denoising
Image denoising provides a simple and clear problem formulation example. The *_observation model_* is:
$
  z(x) = y(x) + eta(x)
$
where:
- $z(x)$ is noisy observation at pixel coordinate $x$
- $y(x)$ is ideal (noise-free) image
- $eta(x)$is the noise component
- $x$ is pixel coordinate
This formulation assumes _additive white Gaussian noise_(AWGN).

We assume that the noise is:
- *Additive Gaussian noise*: $eta(x) ~ N(0, sigma^2)$.
- *Independent and identically distributed (i.i.d.)*: Noise realizations at different pixels are independent.

While real-world noise may exhibit correlations or non-Gaussian characteristics, the AWGN model remains prevalent for algorithm design. Practical systems often transform raw sensor data to approximate this model. The denoising objective is formalized as finding an estimator $hat(y)$ minimizing the the mean squared error (MSE):
$
  "MSE"(hat(bold(y))) = EE[norm(hat(y) - y)^2_2]
$

The observation model provides a *prior on noise* but we also need a *prior on images*.
#definition("Noise Prior")[
  A *noise prior* refers to the assumed statistical properties or characteristics of the noise itself that is corrupting the image. Knowing how the noise behaves helps in modeling and subsequently removing it. For AWGN, the true image $y$ is likely to be a in a circular neighborhood around the observation $z(x)$.
]
#definition("Image Prior")[
  An *image prior* is a statistical model that captures the expected structure of natural images. It is used to regularize the denoising process, guiding the estimator towards plausible solutions.
]

#pagebreak()

== Local Constancy Prior
Assumption: Images are locally constant within small patches. For a constant signal corrupted by Gaussian noise:
$
  hat(y) = 1 / M sum_(i=1)^M z(x_i)
$
Properties of the this estimator:
- *Unbiased:* $EE[hat(y)] = y$ (true signal)
- *Reduced Variance:* $"Var"[hat(y)] = sigma^2 / M$

*Limitations:* Local averaging introduces _bias_ at edges:
#nonum(
  $
    "Bias" = abs(EE[hat(y)] - y) >> 0 "at discontinuities"
  $,
)

#figure(
  canvas({
    import draw: *

    set-style(
      mark: (fill: black, scale: 1),
      stroke: (thickness: 0.4pt, cap: "round"),
    )
    // Axes
    line((0, 0), (10, 0), mark: (end: "stealth"))
    content((10.2, 0), [$x$], anchor: "west")

    line((0, 0), (0, 5), mark: (end: "stealth"))
    content((0, 5.2), [Intensity], anchor: "south")

    // Step function (thick line)
    set-style(stroke: (thickness: 1pt))
    line((0, 1), (5, 1))
    line((5, 1), (5, 3))
    line((5, 3), (10, 3))

    // Reset stroke style
    set-style(stroke: (thickness: 1pt))

    // Labels for flat regions
    content((2.5, 0.5), [Flat region], anchor: "north")
    content((7.5, 0.5), [Flat region], anchor: "north")

    // Dashed vertical lines
    line((2.5, 1), (2.5, 3.5), stroke: (paint: blue, thickness: 1pt, dash: "dashed"))
    line((7.5, 1), (7.5, 3.5), stroke: (paint: red, thickness: 1pt, dash: "dashed"))

    // Noise points (blue circles)
    for x in (0.5, 1.5, 2.5, 3.5, 4.5) {
      circle((x, 1), radius: 0.1, fill: blue.transparentize(70%), stroke: none)
    }

    // Noise points (red circles)
    for x in (5.5, 6.5, 7.5, 8.5, 9.5) {
      circle((x, 3), radius: 0.1, fill: red.transparentize(70%), stroke: none)
    }

    // Edge annotation with arrow
    line((4.8, 1.2), (5.2, 2.8), mark: (start: "stealth", end: "stealth"))
    content((4.6, 2), [Edge], anchor: "east")

    // Top annotations
    content((2.5, 4), [Unbiased estimation], anchor: "south")
    content((7.5, 4), [Biased estimation], anchor: "south")
  }),
  caption: [Bias-variance tradeoff in local averaging],
) <fig:bias_tradeoff>

== Sparsity-Based Image Prior
=== Motivation for Sparsity
Natural images have *sparse representations* in certain transform domains (e.g., DCT), as evidenced by the success of JPEG compression. *Key Insight:* If images can be sparsely represented for compression, this same property can be leveraged for denoising.

=== DCT-Based Denoising Pipeline
*Step 1 : Analysis*
$
  X = D^T S
$
where:
- $S$ is the vectorized image patch
- $D$ is the DCT basis matrix
- $X$ is the DCT coefficients vector

*Step 2: Enforce Sparsity (Thresholding)*
$
  hat(X)_i = cases(
    X_i space space "if" abs(X_i) >= gamma,
    0 space space "if" abs(X_i) < gamma
  )
$
*Important:* Apply thresholding only to the coefficients $i gt.eq 1$ (preserve the DC component).
\
*Step 3: Synthesis*
$
  hat(S) = D hat(X)
$

=== Threshold Selection
#theorem("Universal Thresholding")[
  For AWGN with variance $sigma^2$, the optimal threshold for denoising is given by:
  $
    gamma = sigma sqrt(2 log(n))
  $
  where:
  - $sigma$ is the noise standard deviation
  - $n$ is the dimension of coefficients vector
  For $8 times 8$ patches: $gamma approx 3 sigma$
]


== Noise Standard Deviation Estimation
To use the universal thresholding, we need to estimate $sigma$ from the noisy image itself.

#theorem("Robust Estimation of Noise Standard Deviation")[
  Given a noisy image $Z$, the noise standard deviation can be estimated using the following robust method:
  $
    hat(sigma) = "MAD" / (0.6745 times sqrt(2))
  $
  where MAD = Median Absolute Deviation, defined as:
  $
    "MAD"(D) = "median"(|D - "median"(D)|)
  $
  where $D$ denotes the horizontal differences of the image.
]

== Sliding DCT Algorithm
=== Non-Overlapping Tiles (No Aggregation)
The simplest approach processes the image in non-overlapping $8 times 8$ blocks. Let $Z$ be the noisy image of size $M times N$, partitioned into non-overlapping blocks $B_(i,j)$ of size $8 times 8$. For each block:

$
  X_(i,j) = "DCT"_2(B_(i,j))
$

Apply hard thresholding with universal threshold $gamma = 3sigma$:
$
  hat(X)_(i,j)(u,v) = cases(
    X_(i,j)(u,v) quad & "if " abs(X_(i,j)(u,v)) >= gamma,
    0 quad & "otherwise"
  )
$

Reconstruct each block:
$
  hat(S)_(i,j) = "IDCT"_2(hat(X)_(i,j))
$

The final denoised image is the union of all processed blocks:
$
  hat(Y) = union.big_(i,j) hat(B)_(i,j)
$

*Properties:*
- Complexity: $O(N)$ operations
- Blocking artifacts due to independent processing
- Fast but lower quality

=== Sliding Window with Uniform Weights
For overlapping patches, each pixel receives multiple estimates that must be aggregated.

For each patch position $(i,j)$ with step size $"STEP" = 1$, extract $p times p$ patch $S_(i,j)$ from the noisy image and do the same DCT processing as before, getting the reconstructed patch $hat(S)_(i,j)$.

The denoised image $hat(I)$ is obtained by weighted aggregation:

$
  hat(I)(m,n) = (sum_(i,j in Omega_(m,n)) w dot hat(S)_(i,j)(m-i, n-j)) / (sum_(i,j in Omega_(m,n)) w + epsilon)
$

where $w = 1.0$ is the uniform weight, $epsilon = 10^(-8)$ prevents division by zero, and $Omega_(m,n)$ denotes the set of patch positions $(i,j)$ that contain pixel $(m,n)$.

For an image of size $M times N$, each pixel $(m,n)$ can be covered by at most $p times p$ overlapping patches when using unit step size, but the actual number depends on the pixel's position relative to image boundaries.

=== Sparsity-Adaptive Weight Aggregation
For overlapping patches, each pixel receives multiple estimates that must be aggregated with weights adapted to the sparsity of the thresholded DCT coefficients.

For each patch position $(i,j)$ with step size $"STEP" = 1$, extract $p times p$ patch $S_(i,j)$ from the noisy image, getting the reconstructed patch $hat(S)_(i,j)$.

The sparsity-adaptive weight for each patch is computed based on the number of *non-zero coefficients* after thresholding:

$
  w_(i,j) = "nnz"(hat(X)_(i,j))
$

where $"nnz"(hat(X)_(i,j))$ counts the number of non-zero elements in the thresholded DCT coefficient matrix.

The denoised image $hat(I)$ is obtained by sparsity-weighted aggregation:

$
  hat(I)(m,n) = (sum_(i,j in Omega_(m,n)) w_(i,j) dot hat(S)_(i,j)(m-i, n-j)) / (sum_(i,j in Omega_(m,n)) w_(i,j) + epsilon)
$

where $epsilon = 10^(-8)$ prevents division by zero, and $Omega_(m,n)$ denotes the set of patch positions $(i,j)$ that contain pixel $(m,n)$.

#pagebreak()

== Wiener Filter
The *Wiener filter* is a powerful tool for image denoising, particularly when the noise characteristics are known. It operates in the frequency domain, leveraging the DCT coefficients to perform adaptive filtering based on local statistics.

=== Empirical Wiener Filter
Let $hat(bold(y))^"HT"$ be the *hard threshold estimate*, with DCT coefficients:
$
  hat(bold(x))^"HT" = D^T hat(bold(y))^"HT"
$
The empirical Wiener filter attenuates the DCT coefficients as:
$
  hat(x)^"Wie"_i = (hat(x)^"HT"_i)^2 / ((hat(x)^"HT"_i)^2 + sigma^2) x_i
$
The empirical Wiener estimate is thus:
$
  hat(bold(y))^"HT" = D hat(bold(x))^"Wie"
$

=== Transform Domain Patch Processing

Given an image $bold(Y)$ of size $M times M$, we extract overlapping patches $bold(P)_(i,j)$ of size $p times p$ centered at pixel $(i,j)$:
$
  bold(P)_(i,j) = bold(Y)[i-floor(p/2) : i+floor(p/2), j-floor(p/2) : j+floor(p/2)]
$

For each patch $bold(P)_(i,j)$, we apply the following procedure:
+ *Vectorization*: Convert patch to vector $bold(p)_(i,j) in RR^(p^2)$
+ *Transformation*: Apply orthogonal transform $tilde(bold(p))_(i,j) = bold(T)bold(p)_(i,j)$
+ *Preliminary Estimation*: Obtain initial estimate $hat(tilde(bold(p)))_(i,j)^((0))$ using a simple denoising method
+ *Wiener Filtering*: Apply empirical Wiener filter coefficient-wise
+ *Inverse Transform*: Reconstruct patch $hat(bold(p))_(i,j) = bold(T)^(-1)hat(tilde(bold(p)))_(i,j)$


#pagebreak()

= Limitations of Sparsity-Based Denoising
== The Sparsity Problem
While orthonormal bases provide computational convenience and guarantee unique representations, they suffer from a fundamental limitation: _no single orthonormal basis can provide sparse representations for all signals of interest_.

#example("DCT Basis Limitation")[
  Consider a signal $bold(s)_0 in RR^n$ that admits a sparse representation with respect to the Discrete Cosine Transform (DCT) basis $bold(D)_"DCT"$:

  $ bold(s_0) = bold(D)_"DCT" bold(x_0) $

  where $bold(x_0)$ is sparse (most entries are zero).

  Now consider the modified signal:

  $ bold(s) = bold(s_0) + lambda bold(e_j) $

  where $bold(e_j)$ is the $j$-th canonical basis vector and $lambda in RR$ is a scaling factor.

  The DCT representation of $bold(s)$ becomes:

  $
    bold(x) = bold(D)_"DCT"^T bold(s) = bold(D)_"DCT"^T bold(s_0) + lambda bold(D)_"DCT"^T bold(e_j) = bold(x_0) + lambda bold(D)_"DCT"^T bold(e_j)
  $

  Since $bold(D)_"DCT"^T bold(e_j)$ is typically dense (all entries are non-zero), the addition of a single spike destroys the sparsity of the representation.
]

== Experimental Demonstration

// NOTE: This section would typically include figures showing the DCT coefficients before and after adding a spike

The experimental verification of this limitation involves:

1. Generate a sparse signal $bold(s_0)$ with respect to DCT basis
2. Add a single spike: $bold(s) = bold(s_0) + lambda bold(e_j)$
3. Compute DCT coefficients of both signals
4. Observe the loss of sparsity in the modified signal

The results consistently show that the addition of a single spike causes all DCT coefficients to become significant, effectively destroying the sparse structure that denoising algorithms rely upon.

== Overcomplete Dictionaries: The Solution
=== Motivation for Redundancy
The solution to the sparsity limitation lies in abandoning the constraint of orthonormality and embracing redundancy. Instead of using a single $n times n$ orthonormal basis, we construct an $n times m$ dictionary matrix $bold(D)$ where $m > n$.

#definition("Overcomplete Dictionary")[
  An _overcomplete dictionary_ is a matrix $bold(D) in RR^(n times m)$ with $m > n$ such that:

  $ "span"{bold(d)_1, bold(d)_2, ..., bold(d)_m} = RR^n $

  where $bold(d)_i$ are the columns of $bold(D)$.
]

=== Construction of Overcomplete Dictionaries
For the DCT-spike example, we construct the overcomplete dictionary by concatenating the DCT basis with the *canonical basis*:

$ bold(D) = mat(bold(D)_"DCT", bold(I)) in RR^(n times 2n) $

This construction ensures that:
- Signals sparse in DCT domain remain sparse
- Signals sparse in canonical domain remain sparse
- Mixed signals (DCT-sparse + spikes) admit sparse representations

#example("Sparse Representation with Overcomplete Dictionary")[
  Consider the signal $bold(s) = bold(s)_0 + lambda bold(e)_j$ where $bold(s)_0 = bold(D)_"DCT" bold(x)_0$ with sparse $bold(x)_0$.

  The representation with respect to the overcomplete dictionary is:

  $ bold(s) = bold(D) mat(bold(x)_0; lambda bold(e)_j) $

  The coefficient vector $mat(bold(x)_0; lambda bold(e)_j) in RR^(2n)$ is sparse, containing only the non-zero entries of $bold(x)_0$ plus the single entry $lambda$ at position $j$ in the second block.
]

=== Theoretical Properties of Overcomplete Systems

#theorem("Rouché-Capelli Theorem")[
  Consider the linear system $bold(D)bold(x) = bold(s)$ where $bold(D) in RR^(n times m)$ and $bold(s) in RR^n$. The system admits a solution if and only if:

  $ "rank"(bold(D)) = "rank"(mat(bold(D), bold(s))) $
] <thm:rouche>

When $m > n$ and $"rank"(bold(D)) = n$, the system has infinitely many solutions forming an affine subspace of dimension $m - n$.

#theorem("Solution Space Dimension")[
  If $bold(D) in RR^(n times m)$ with $m > n$ and $"rank"(bold(D)) = n$, then for any $bold(s) in RR^n$, the solution set of $bold(D)bold(x) = bold(s)$ forms an affine subspace of dimension $m - n$.
]

#pagebreak()

== Regularization and Sparse Recovery
=== The Ill-Posed Nature of Overcomplete Systems
The abundance of solutions in overcomplete systems necessitates additional criteria for solution selection. This is where regularization theory becomes essential.

#definition("Regularization")[
  Given an *ill-posed* problem $bold(D)bold(x) = bold(s)$ with multiple solutions, _regularization_ involves solving:

  $ hat(bold(x)) = arg min_(bold(x)) J(bold(x)) quad "subject to" quad bold(D)bold(x) = bold(s) $

  where $J: RR^m -> RR_+$ is a regularization functional encoding our prior knowledge about the desired solution.
]

=== $ell_2$ Regularization: Ridge Regression

The most mathematically tractable regularization is the $ell_2$ norm:

$ J(bold(x)) = 1/2 ||bold(x)||_2^2 = 1/2 sum_(i=1)^m x_i^2 $

This leads to the constrained optimization problem:

$ hat(bold(x)) = arg min_(bold(x)) 1/2 ||bold(x)||_2^2 quad "subject to" quad bold(D)bold(x) = bold(s) $

Alternatively, we can formulate the unconstrained version:

$ hat(bold(x)) = arg min_(bold(x)) 1/2 ||bold(D)bold(x) - bold(s)||_2^2 + lambda/2 ||bold(x)||_2^2 $

=== Analytical Solution via Matrix Calculus

The unconstrained $ell_2$ regularization problem admits a closed-form solution. To derive this, we use matrix calculus.

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
    &= 1/2 vec(x)^T bold(D)^T bold(D) vec(x) - vec(s)^T bold(D) vec(x) + 1/2 vec(s)^T vec(s) + lambda/2 vec(x)^T vec(x)
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
