#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *

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