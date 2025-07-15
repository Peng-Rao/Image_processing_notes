#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *

= Image Prior
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
