#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *

= Discrete Cosine Transform (DCT)
== 1D DCT
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

== 2D DCT
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

== DCT-Based Denoising Pipeline
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

#pagebreak()

