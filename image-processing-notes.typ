#import "@local/simple-note:0.0.1": *
#show: codly-init.with()

#show: simple-note.with(
  title: [ Image Processing ],
  date: datetime(year: 2025, month: 2, day: 17),
  authors: (
    (
      name: "Rao",
      github: "https://github.com/Raopend",
      homepage: "https://github.com/Raopend",
    ),
  ),
  affiliations: (
    (
      id: "1",
      name: "Politecnico di Milano",
    ),
  ),
  // cover-image: "./figures/polimi_logo.png",
  background-color: "#FAF9DE",
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
  image("figures/enforce-sparsity.jpg", width: 80%),
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
We assume that the noise is:
- *Additive Gaussian noise*: $eta(x) ~ N(0, sigma^2)$.
- *Independent and identically distributed (i.i.d.)*: Noise realizations at different pixels are independent.
Our goal is to estimate $hat(y)$ that is close to the true image $y$.

The observation model provides a *prior on noise* but we also need a *prior on images*.
- *Noise Prior:* Given the Gaussian assumption, the true image $y$ is likely to be a in a circular neighborhood around the observation $z(x)$.
- *Image Prior:* Additional assumptions about the structure of natural images are needed for effective denoising.

== Simple Prior: Local Constancy
Assumption: Images are locally constant within small patches. For a constant signal corrupted by Gaussian noise:
$
  hat(y) = 1 / M sum_(i=1)^M z(x_i)
$
Properties of the this estimator:
- *Unbiased:* $EE[hat(y)] = y$ (true signal)
- *Reduced Variance:* $"Var"[hat(y)] = sigma^2 / M$
- *Limitation:* Fails at edges and discontinuities

== Sparsity-Based Image Prior
=== Motivation for Sparsity
Natural images have *sparse representations* in certain transform domains (e.g., DCT), as evidenced by the success of JPEG compression.

*Key Insight:* If images can be sparsely represented for compression, this same property can be leveraged for denoising.

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

=== Universal Denoising (Donoho)
*Extreme Value Theory*
$
  gamma = sigma sqrt(2 log(n))
$
where:
- $sigma$ is the noise standard deviation
- $n$ is the dimension of coefficients vector
For $8 times 8$ patches: $gamma approx 3 sigma$

== Noise Standard Deviation Estimation
To use the universal thresholding, we need to estimate $sigma$ from the noisy image itself.

=== Robust Estimation Method
*Step 1: Compute Image Differences*
$
  D = Z * [-1, 1]
$
This computes differences between adjacent pixels.
\
*Step 2: Robust Standard Deviation Estimation*
$
  sigma = "MAD"(D) / (0.7676 times sqrt(2))
$
where MAD = Median Absolute Deviation, defined as:
$
  "MAD"(D) = "median"(|D - "median"(D)|)
$
*Rationale:*
- In flat regions: $D approx eta_i - eta_j$
- Robust estimator ignores outliers from edges
- Magic number $0.7676$ normalizes MAD to standard deviation for Gaussian distributions

== Sliding DCT Algorithm
Processing patches independently creates artifacts at patch boundaries, especially when patches contain edges.

The solution is overlapping patches, where each pixel is processed multiple times.

+ For each pixel $(r, c)$ in the image:
  + Extract patch $S$ centered at $(r, c)$ with size $8 times 8$.
  + Apply DCT denoising pipeline: $S arrow X arrow hat(X) arrow hat(S)$
  + Store estimate for center pixel
+ Aggregate all estimates for each pixel

Instead of simple averaging, use weights based on sparsity of coefficients:
$
  w = 1 / "Number of non-zero coefficients"
$
Rationale: Sparser representations are more reliable under our prior assumption. So the final estimate is:
$
  hat(y)(r, c) = (sum_"patches" w hat(S)) / (sum_"patches" w)
$

=== Advantages
- Translation invariant (due to sliding window)
- Adaptive filtering (different processing for different patches)
- Handles edges better than simple smoothing
- Leverages natural image statistics

=== Limitations
- Computational cost (processing every pixel)
- DCT may not be optimal basis for all image patches
- Border effects (fewer estimates at image boundaries)
