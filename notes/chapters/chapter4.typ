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