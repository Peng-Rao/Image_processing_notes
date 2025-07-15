#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "../template.typ": *
= Introduction to Sparsity
The principle of sparsity or "parsimony" consists in representing some phenomenon with as few variable as possible. Stretch back to philosopher William Ockham in the 14th century, Wrinch and Jeffreys relate simplicity to parsimony:

The *existence of simple laws* is, then, apparently, to be regarded as *a quality of nature*; and accordingly we may infer that it is justifiable to *prefer a simple law to a more complex one that fits our observations slightly better*.

== Sparsity in Statistics
Sparsity is used to *prevent overfitting and improve interpretability of learned models*. In model fitting, the number of parameters is typically used as a criterion to perform model selection. See Bayes Information Criterion (BIC), Akaike Information Criterion (AIC), ...., Lasso.

== Sparsity in Signal Processing
*Signal Processing*: similar concepts but different terminology. *Vectors corresponds to signals* and data modeling is crucial for performing various operations such as *restoration, compression, solving inverse problems*.

Signals are approximated by sparse linear combinations of *prototypes*(basis elements / atoms of a dictionary), resulting in simpler and compact model.

#figure(
  image("../figures/enforce-sparsity.png"),
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
