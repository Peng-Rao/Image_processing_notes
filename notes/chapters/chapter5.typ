#import "@local/simple-note:0.0.1": *
= Dictionary Learning
Dictionary learning represents a fundamental paradigm in signal processing and machine learning, where the objective is to discover optimal sparse representations of data. Unlike traditional approaches that rely on pre-constructed bases such as the Discrete Cosine Transform (DCT) or Principal Component Analysis (PCA), dictionary learning adapts the representation to the specific characteristics of the training data.

The concept of dictionary learning emerged from the intersection of sparse coding theory and matrix factorization techniques. While classical orthogonal transforms like DCT and PCA provide optimal representations for specific signal classes, they often fail to capture the intrinsic structure of complex, real-world data.

== Problem Formulation
#definition("Dictionary Learning")[
  Given a set of training signals $bold(Y) = [bold(y)_1, bold(y)_2, ..., bold(y)_N] in RR^(n)$, the dictionary learning problem seeks to find:
  - A dictionary matrix $bold(D) in RR^(n times m)$ with $m > n$ (overcomplete)
  - Sparse coefficient vectors $bold(x)_1, bold(x)_2, ..., bold(x)_N in RR^(m)$
]

#pagebreak()
