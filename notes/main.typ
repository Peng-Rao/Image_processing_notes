#import "@local/simple-note:0.0.1": *
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "@preview/ctheorems:1.1.3": *
#show: thmrules

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
#let definition = thmbox("definition", "Definition", inset: (x: 0em, top: 0em))
#let proposition = thmbox("proposition", "Proposition", inset: (x: 0em, top: 0em))
#let theorem = thmbox("theorem", "Theorem", inset: (x: 0em, top: 0em))
#let lemma = thmbox("lemma", "Lemma", inset: (x: 0em, top: 0em))
#set math.mat(delim: "[")
#set math.vec(delim: "[")
#set math.equation(supplement: [Eq.])

#let nonum(eq) = math.equation(block: true, numbering: none, eq)
#let firebrick(body) = text(fill: rgb("#b22222"), body)

= Introduction to Sparsity
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

= Dictionary Learning
== Introduction to Dictionary Learning
Dictionary learning represents a fundamental paradigm in signal processing and machine learning, where the objective is to discover optimal sparse representations of data. Unlike traditional approaches that rely on pre-constructed bases such as the Discrete Cosine Transform (DCT) or Principal Component Analysis (PCA), dictionary learning adapts the representation to the specific characteristics of the training data.

The concept of dictionary learning emerged from the intersection of sparse coding theory and matrix factorization techniques. While classical orthogonal transforms like DCT and PCA provide optimal representations for specific signal classes, they often fail to capture the intrinsic structure of complex, real-world data.

#definition("Dictionary Learning Problem")[
  Given a set of training signals $bold(y)_1, bold(y)_2, ..., bold(y)_N in RR^n$, the dictionary learning problem seeks to find:
  + A dictionary matrix $bold(D) in RR^{n times m}$ with $m > n$ (redundant dictionary)
  + Sparse coefficient vectors $bold(x)_1, bold(x)_2, ..., bold(x)_N in RR^m$

  such that $bold(y)_i approx bold(D) bold(x)_i$ for all $i = 1, 2, ..., N$, where each $bold(x)_i$ has at most $T_0$ non-zero entries.
]

== Problem Formulation
Let $bold(Y) = [bold(y)_1, bold(y)_2, ..., bold(y)_N] in RR^(n times N)$ denote the training matrix, where each column represents a training signal. Similarly, let $bold(X) = [bold(x)_1, bold(x)_2, ..., bold(x)_N] in RR^(m times N)$ represent the sparse coefficient matrix.

The dictionary learning problem can be formulated as the following optimization:

$
  min_(bold(D), bold(X)) norm(bold(Y) - bold(D) bold(X))_F^2
  quad "subject to" quad norm(bold(x)_i)_0 <= T_0, quad forall i = 1, 2, ..., N
$

where $norm(dot)_F$ denotes the Frobenius norm and $norm(dot)_0$ is the $ell_0$ pseudo-norm counting non-zero entries.


*(Normalization Constraint:)*. To resolve scaling ambiguities, we impose the constraint that each column of $bold(D)$ has unit $ell_2$ norm:
$
  norm(bold(d)_j)_2 = 1, quad forall j = 1, 2, ..., m
$


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

== Matrix Spark
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
