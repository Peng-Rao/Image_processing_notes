#import "@local/simple-note:0.0.1": *
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
