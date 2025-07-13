= Sparsity and Parsimony
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

#pagebreak()