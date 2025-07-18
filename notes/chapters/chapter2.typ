#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *
#import "@preview/algorithmic:1.0.2"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

= Image Prior
== Image Denoising Problem
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

In this chapter, we will explore various image priors and their applications in denoising. The goal is to leverage the statistical properties of natural images to improve the quality of denoised outputs.

#pagebreak()

== Local Constancy Prior
Assumption: Images are locally constant within small patches. For a constant signal corrupted by Gaussian noise:
$
  hat(y) = 1 / M sum_(i=1)^M z(x_i)
$
Properties of the this estimator:
- *Unbiased:* $EE[hat(y)] = y$ (true signal)
- *Reduced Variance:* $"Var"[hat(y)] = sigma^2 / M$

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

Consider the classical approach of estimating pixel intensities through local averaging. For a pixel at location $(i,j)$, a naive estimate would be:
$
  hat(x)_(i,j) = 1/(|U_(i,j)|) sum_((k,l) in U_(i,j)) y_(k,l)
$ <simple_average>
where $U_(i,j)$ denotes a neighborhood around pixel $(i,j)$, and $|U_(i,j)|$ is its cardinality.

This approach is equivalent to convolution with a normalized box kernel:
$
  hat(x) = y * h
$
where $h$ is the box kernel with support $U_(i,j)$.

The primary limitation of local averaging becomes apparent near image discontinuities. Consider an ideal edge scenario where:

$
  x_(i,j) = cases(
    a & "if " (i,j) in Omega_1,
    b & "if " (i,j) in Omega_2
  )
$
where $Omega_1$ and $Omega_2$ are disjoint regions separated by an edge, and $a != b$.

For a pixel $(i,j)$ near the edge boundary, the local average becomes:
$
  hat(x)_(i,j) &= 1/(|U_(i,j)|) (sum_((k,l) in U_(i,j) inter Omega_1) y_(k,l) + sum_((k,l) in U_(i,j) inter Omega_2) y_(k,l)) \
  &approx (|Omega_1 inter U_(i,j)|)/(|U_(i,j)|) a + (|Omega_2 inter U_(i,j)|)/(|U_(i,j)|) b
$
This results in a blurred edge transition, motivating the need for adaptive filtering strategies.


#pagebreak()

== Non-local Means Prior
The Non-Local Means (NLM) algorithm addresses the limitations of local averaging by exploiting the _self-similarity_ property of natural images. *The key insight is that similar patches exist throughout the image, not just in local neighborhoods.*

#definition("Self-Similarity Prior")[
  For a natural image $x$, there exist multiple patches ${P_k}$ such that:
  $ ||P_i - P_j||_2 < epsilon $
  for some small threshold $epsilon > 0$, where $P_k$ represents a patch extracted from the image.
] <self_similarity>

The NLM estimate for pixel $(i,j)$ is given by:
$
  hat(x)_(i,j) = sum_((k,l) in S_(i,j)) w_(i,j)(k,l) dot y_(k,l)
$ <nlm_estimate>
where $S_(i,j)$ represents the search window around pixel $(i,j)$, and $w_(i,j)(k,l)$ are the similarity weights.

=== Weight Computation
The similarity weights are computed based on patch distances:
$
  w_(i,j)(k,l) = 1/(Z_(i,j)) exp(-(d^2(P_(i,j), P_(k,l)))/(h^2))
$ <nlm_weights>
where:
- $P_(i,j)$ and $P_(k,l)$ are patches centered at $(i,j)$ and $(k,l)$ respectively
- $d^2(P_(i,j), P_(k,l))$ is the squared Euclidean distance between patches
- $h$ is the filtering parameter controlling the decay rate
- $Z_(i,j)$ is the normalization constant ensuring $sum w_(i,j)(k,l) = 1$

=== Patch Distance Computation
The patch distance is computed as:
$
  d^2(P_(i,j), P_(k,l)) = 1/(|P|) sum_((u,v) in P) |y_(i+u,j+v) - y_(k+u,l+v)|^2
$ <patch_distance>
where $P$ represents the patch domain and $|P|$ is the number of pixels in the patch.

*Noise-Aware Distance*
In the presence of noise, the distance can be corrected as:
$ d^2_"corrected"(P_(i,j), P_(k,l)) = max(d^2(P_(i,j), P_(k,l)) - 2sigma^2, 0) $ <noise_corrected_distance>
This correction accounts for the noise contribution to the patch distance.

=== Normalization and Properties
The normalization constant is given by:
$ Z_(i,j) = sum_((k,l) in S_(i,j)) exp(-(d^2(P_(i,j), P_(k,l)))/(h^2)) $ <normalization>

#theorem("NLM Consistency")[
  For a noiseless image, the NLM estimate satisfies:
  $ lim_(sigma -> 0) hat(x)_(i,j) = x_(i,j) $
] <nlm_consistency>


*Proof:*
As $sigma -> 0$, the patch distances approach their true values, and the weight distribution becomes increasingly concentrated around patches identical to the reference patch.

=== Algorithm Framework
#algorithm-figure("Non-Local Means Algorithm", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([NLM], ([Noisy image $y$], [patch size $p$], [search window size $s$], [filtering parameter $h$]), {
    Comment[*Input:* Noisy image $y$, patch size $p$, search window size $s$, filtering parameter $h$]
    Comment[*Output:* Denoised image $hat(x)$]
    LineBreak
    Comment[For each pixel $(i,j)$ in the image:]
    For([each pixel $(i,j)$ in the image], {
      Comment[Initialize weight matrix $W = 0$]
      Assign[$W$][$0$]
      LineBreak
      Comment[Define search window $S_(i,j)$ of size $s times s$]
      Assign[$S_(i,j)$][search window of size $s times s$ around $(i,j)$]
      LineBreak
      Comment[For each pixel $(k,l)$ in $S_(i,j)$:]
      For([each pixel $(k,l)$ in $S_(i,j)$], {
        Comment[Extract patches $P_(i,j)$ and $P_(k,l)$ of size $p times p$]
        Assign[$P_(i,j), P_(k,l)$][patches of size $p times p$]
        LineBreak
        Comment[Compute distance $d^2(P_(i,j), P_(k,l))$ using patch distance formula]
        LineBreak
        Assign[$d^2(P_(i,j), P_(k,l))$][$1/abs(P) sum_((u,v) in P) abs(y_(i+u,j+v) - y_(k+u,l+v))^2$]
        LineBreak
        Comment[Compute weight $w_(i,j)(k,l)$ using NLM weights formula]
        LineBreak
        Assign[$w_(i,j)(k,l)$][$exp(-(d^2(P_(i,j), P_(k,l)))/h^2)$]
        LineBreak
        Assign[$W(k,l)$][$w_(i,j)(k,l)$]
      })
      LineBreak
      Comment[Normalize weights: $W = W / sum W$]
      Assign[$W$][$W / sum W$]
      LineBreak
      Comment[Compute estimate: $hat(x)_(i,j) = sum_((k,l)) W(k,l) dot y_(k,l)$]
      Assign[$hat(x)_(i,j)$][$sum_((k,l)) W(k,l) dot y_(k,l)$]
    })
    LineBreak
    Return[$hat(x)$]
  })
})

Near image boundaries, patches may extend beyond the image domain. Common strategies include:
+ *Symmetric Padding*: Reflect image content across boundaries
+ *Periodic Padding*: Wrap image content periodically
+ *Zero Padding*: Extend with zeros (not recommended)

The symmetric padding approach is preferred as it maintains image statistics:
$
  tilde(x)_(i,j) = cases(
    x_(i,j) & "if " (i,j) "is inside image",
    x_(2N-i,j) & "if " i > N " (bottom boundary)",
    x_(i,2M-j) & "if " j > M " (right boundary)"
  )
$

=== Parameter Selection
==== Typical Parameter Values
- Patch size: $p = 7 times 7$ (provides good balance between detail and computational cost)
- Search window: $s = 21 times 21$ (sufficient for finding similar patches)
- Filtering parameter: $h = 10sigma$ (empirically determined)

==== Adaptive Parameter Selection
The filtering parameter can be adapted based on local image characteristics:
$
  h_(i,j) = alpha dot sigma dot sqrt("LocalVariance"(P_(i,j)))
$
where $alpha$ is a scaling factor and $"LocalVariance"$ measures local image activity.

== Block-Matching 3D (BM3D) Algorithm
The Block-Matching 3D (BM3D) algorithm extends the self-similarity concept by combining it with sparsity-based denoising. The key innovations include:
+ *Grouping*: Collect similar patches into 3D arrays
+ *Collaborative Filtering*: Process groups jointly using 3D transforms
+ *Aggregation*: Combine processed patches back into the image

=== Patch Grouping
For a reference patch $P_(i,j)$, we define the group $G_(i,j)$ as:
$ G_(i,j) = {P_(k,l) : d^2(P_(i,j), P_(k,l)) < tau} $ <group_definition>
where $tau$ is a similarity threshold.

=== 3D Array Construction
The group is arranged as a 3D array $cal(G) in bb(R)^(p times p times K)$, where $K = |G_(i,j)|$ is the number of patches in the group.

The BM3D algorithm applies a 3D transform to exploit both spatial and inter-patch correlations:
$
  cal(T) = cal(W)_(3D) dot cal(G)
$ <3d_transform_eq>
where $cal(W)_(3D)$ represents the 3D transform operator.

=== Separable Transform
The 3D transform is typically implemented as a separable transformation:
$
  cal(T) & = cal(W)_1 dot (cal(W)_2 dot (cal(W)_3 dot cal(G)))   \
         & = (cal(W)_1 times cal(W)_2 times cal(W)_3) dot cal(G)
$
where $cal(W)_1$ and $cal(W)_2$ are 2D transforms (e.g., DCT) applied to each patch, and $cal(W)_3$ is a 1D transform applied across the grouping dimension.

=== Hard Thresholding Stage
In the first stage, BM3D applies hard thresholding to the transform coefficients:
$
  hat(cal(T)) = cal(H)_lambda (cal(T))
$ <hard_threshold>
where the hard thresholding operator is defined as:
$
  cal(H)_lambda (t) = cases(
    t & "if " |t| > lambda,
    0 & "if " |t| <= lambda
  )
$

=== Threshold Selection
The threshold is typically chosen as:
$
  lambda = beta dot sigma
$
where $beta$ is a parameter controlling the aggressiveness of denoising (typically $beta = 2.7$).

=== Collaborative Filtering Stage
The second stage performs collaborative filtering using both the noisy and first-stage estimates:
$
  hat(cal(T))^((2)) = (|cal(T)^((1))|^2)/(|cal(T)^((1))|^2 + sigma^2) dot cal(T)^((0))
$ <collaborative_filter>
where:
- $cal(T)^((0))$ represents the transform of the noisy group
- $cal(T)^((1))$ represents the transform of the first-stage estimate
- $|dot|^2$ denotes element-wise squared magnitude

=== Aggregation and Weighting
After processing, patches must be aggregated back into the image. Due to overlapping patches, multiple estimates exist for each pixel:

$ hat(x)_(i,j) = (sum_(G in.rev (i,j)) w_G dot hat(x)_(i,j)^((G)))/(sum_(G in.rev (i,j)) w_G) $ <aggregation_eq>

where the sum is over all groups $G$ containing pixel $(i,j)$.

*Sparsity-Aware Weighting*

The weights are chosen based on the sparsity of the processed group:

$ w_G = 1/(||cal(T)_G||_0) $ <sparsity_weight>

where $||dot||_0$ denotes the number of non-zero coefficients.


#pagebreak()

== Sparsity-Based Image Prior
Natural images have *sparse representations* in certain *transform domains* (e.g., DCT), as evidenced by the success of JPEG compression. There are two types of sparsity models:
- *Transform-domain sparsity*: Images can be represented as sparse linear combinations of basis functions in a specific transform domain.
- *Synthesis Sparse Model*: The synthetic sparse model assumes that a signal (such as an image) can be synthesized through a linear combination of a small number of "atoms." These "atoms" are drawn from a collection known as a *dictionary*.

=== Transform-domain Sparsity
*Transform-domain sparsity* is a fundamental concept in signal and image processing, asserting that many real-world signals and images, though complex in their original form, can be represented much more efficiently in a different mathematical domain. This efficiency is achieved when, after a specific mathematical transformation, the majority of the resulting coefficients are zero or close to zero, leaving only a few significant, non-zero values that capture the essential information of the original data. Some of the most common and powerful transforms include:
- *Discrete Cosine Transform (DCT)*: Widely used in image compression (e.g., JPEG), it transforms spatial domain data into frequency domain, emphasizing low-frequency components.
- *Wavelet Transform*: Decomposes signals into different frequency components, allowing for multi-resolution analysis.
- *Fourier Transform (DFT)*: Converts signals from time domain to frequency domain, revealing periodic structures.

=== Synthesis Sparse Model
The *synthesis sparse model* is a mathematical framework used to represent signals or images as linear combinations of a small number of basis functions, known as "atoms." This model is particularly useful in applications like image compression, denoising, and reconstruction, where the goal is to efficiently represent data while preserving essential features. There are two main approaches to synthesis sparse modeling:
- *Sparse Coding*: In this approach, the signal is expressed as a linear combination of a dictionary of atoms. The goal is to find a sparse representation where only a few coefficients are non-zero, indicating that only a small number of atoms are used to reconstruct the signal. This is often achieved through optimization techniques that minimize the reconstruction error while enforcing sparsity constraints.
- *Dictionary Learning*: This approach involves learning a dictionary of atoms from a set of training signals. The dictionary is optimized to capture the underlying structure of the data, allowing for efficient representation. The learned dictionary can then be used to represent new signals or images in a sparse manner. Dictionary learning algorithms aim to find a set of basis functions that best represent the training data, often through iterative optimization techniques.

#pagebreak()
