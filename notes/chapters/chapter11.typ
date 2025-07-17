#import "@local/simple-note:0.0.1": attention, example, simple-note, zebraw
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart, plot
#import "../template.typ": *
#import "@preview/algorithmic:1.0.2"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

= Multiple Model Fitting
The extension from single-model to multi-model fitting introduces fundamental challenges that require sophisticated approaches. The core difficulty lies in the chicken-and-egg problem: clustering points requires knowledge of models, while fitting models requires knowledge of point clusters.

#definition("Multi-Model Fitting Problem")[
  Given a dataset $cal(D) = {bold(x)_i}_(i=1)^N$ and a model family $cal(M)$, find:
  $ Theta & = {theta_1, theta_2, ..., theta_K} $ <model_set>
  $ cal(P) & = {cal(P)_1, cal(P)_2, ..., cal(P)_K} $ <partition_set>
  where $Theta$ represents the set of model parameters and $cal(P)$ represents a partition of the data such that:
  $ union.big_(k=1)^K cal(P)_k & = cal(D) without cal(O) $ <partition_union>
  $ cal(P)_i inter cal(P)_j & = emptyset quad forall i != j $ <partition_disjoint>
  and each $theta_k$ optimally fits the points in $cal(P)_k$.
] <multimodel_problem>

The fundamental challenge in multi-model fitting can be formalized as follows:
+ *Clustering Requirement*: To fit model $theta_k$, we need to identify the subset $cal(P)_k subset cal(D)$ of points that belong to model $k$
+ *Model Requirement*: To cluster point $bold(x)_i$ into partition $cal(P)_k$, we need to know model $theta_k$ to compute the distance $d(bold(x)_i, theta_k)$

This circular dependency necessitates iterative or simultaneous approaches that can break the cycle through:
- _Sequential strategies_: Fit models one at a time, removing inliers after each fit
- _Simultaneous strategies_: Jointly optimize over all models and partitions
- _Voting strategies_: Use parameter space voting to identify multiple models

== Outlier Complexity in Multi-Model Settings <multimodel_outliers>

In multi-model fitting, the effective outlier ratio increases significantly. Consider fitting model $theta_k$ in the presence of $K$ models:

$ epsilon_"effective" = epsilon_"true" + (K-1)/K (1 - epsilon_"true") $ <effective_outlier_ratio>

where $epsilon_"true"$ is the true outlier ratio and $epsilon_"effective"$ is the effective outlier ratio seen by each individual model.

*Example:* With 10 equally represented models and 10% true outliers:
$ epsilon_"effective" & = 0.1 + 9/10 (1 - 0.1) = 0.1 + 0.9 dot 0.9 = 0.91 $

This dramatic increase in effective outlier ratio severely impacts the performance of standard RANSAC, motivating specialized multi-model approaches.

#pagebreak()

== Sequential Multi-Model Fitting
Sequential approaches address multi-model fitting by iteratively applying single-model fitting techniques, removing inliers after each successful fit. While conceptually simple, these methods can suffer from order dependency and accumulation of errors.

#algorithm-figure("Sequential RANSAC", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([Sequential-RANSAC], ([dataset $cal(D)$], [minimum consensus $tau_"min"$], [maximum models $K_"max"$]), {
    Comment[*Input:* Dataset $cal(D)$, minimum consensus $tau_"min"$, maximum models $K_"max"$]
    Comment[*Output:* Set of models $Theta$]
    LineBreak
    Comment[Initialize:]
    Assign[$cal(D)_"current"$][$cal(D)$]
    Assign[$Theta$][$emptyset$]
    Assign[$k$][$0$]
    LineBreak
    Comment[While $k < K_"max"$ and $abs(cal(D)_"current") >= tau_"min"$:]
    While($k < K_"max" and abs(cal(D)_"current") >= tau_"min"$, {
      Comment[(a) Apply RANSAC to $cal(D)_"current"$ to find model $theta_k$]
      Assign[$theta_k$][$"RANSAC"(cal(D)_"current")$]
      LineBreak
      Comment[(b) Compute consensus set]
      Assign[$cal(C)_k$][${bold(x)_i in cal(D)_"current" : d(bold(x)_i, theta_k) <= tau}$]
      LineBreak
      Comment[(c) If $abs(cal(C)_k) < tau_"min"$, terminate]
      If($abs(cal(C)_k) < tau_"min"$, {
        Comment[Terminate - insufficient consensus]
        Break
      })
      LineBreak
      Comment[(d) Refine $theta_k$ using least squares on $cal(C)_k$]
      Assign[$theta_k$][$"LeastSquares"(cal(C)_k)$]
      LineBreak
      Comment[(e) Add $theta_k$ to $Theta$]
      Assign[$Theta$][$Theta union {theta_k}$]
      LineBreak
      Comment[(f) Update: remove inliers from current dataset]
      Assign[$cal(D)_"current"$][$cal(D)_"current" without cal(C)_k$]
      LineBreak
      Comment[(g) Increment model counter]
      Assign[$k$][$k + 1$]
    })
    LineBreak
    Return[$Theta$]
  })
})

#theorem("Sequential RANSAC Convergence")[
  Under the assumption that models are well-separated and the minimum consensus threshold is appropriately chosen, Sequential RANSAC will find all models with probability:
  $ P_"success" = product_(k=1)^K (1 - (1 - (1-epsilon_k)^m)^(T_k)) $ <sequential_success_prob>
  where $epsilon_k$ is the outlier ratio when fitting model $k$ and $T_k$ is the number of RANSAC iterations for model $k$.
] <sequential_convergence>

*Limitations of Sequential RANSAC:*
+ _Order Dependency_: The sequence in which models are found depends on their relative support and may not reflect the true underlying structure
+ _Error Accumulation_: Misclassified points in early iterations affect subsequent model fitting
+ _Threshold Sensitivity_: Performance heavily depends on the choice of inlier threshold and minimum consensus
+ _Suboptimal Solutions_: Greedy selection may lead to locally optimal but globally suboptimal solutions

Sequential RANSAC can be understood through the lens of preference matrices, which provide insight into simultaneous multi-model approaches.

#definition("Preference Matrix")[
  The preference matrix $bold(P) in bb(R)^(N times M)$ is defined as:
  $
    P_(i j) = cases(
      1 & "if " d(bold(x)_i, theta_j) <= tau,
      0 & "otherwise"
    )
  $ <preference_matrix_binary>
  where $N$ is the number of data points and $M$ is the number of model hypotheses.
] <preference_matrix_def>

Alternatively, the preference matrix can store residuals:
$ P_(i j) = d(bold(x)_i, theta_j) $ <preference_matrix_residual>

*Conceptual Insights:*
- Each row represents a data point's affinity to all model hypotheses
- Each column represents a model's support across all data points
- Sequential RANSAC selects the column with maximum support, then removes corresponding rows
- Simultaneous approaches can exploit the full matrix structure

#pagebreak()

== Simultaneous Multi-Model Fitting
Simultaneous approaches to multi-model fitting attempt to identify all models jointly, avoiding the limitations of sequential methods. The Hough transform represents a classical voting-based approach that has been widely successful in computer vision applications.

Instead of sequentially selecting columns from the preference matrix, simultaneous approaches seek to:
1. _Cluster rows_: Group data points with similar preference patterns
2. _Regularize solutions_: Incorporate sparsity constraints to prefer simpler models
3. _Optimize jointly_: Minimize a global objective function over all models and assignments

#definition("Joint Optimization Problem")[
  The simultaneous multi-model fitting problem can be formulated as:
  $
    min_(Theta, cal(P)) quad & sum_(k=1)^K sum_(bold(x)_i in cal(P)_k) rho(d(bold(x)_i, theta_k)) + lambda ||Theta||_0
  $ <joint_objective>
  $ "subject to" quad & cal(P)_i sect cal(P)_j = emptyset quad forall i != j $ <joint_constraint1>
  $ & union.big_(k=1)^K cal(P)_k subset cal(D) $ <joint_constraint2>
  where $rho(dot)$ is a robust loss function and $lambda$ controls model complexity.
] <joint_optimization>

=== The Hough Transform
The Hough transform provides an elegant solution to multi-model fitting by transforming the point clustering problem into a peak detection problem in parameter space.

The key insight of the Hough transform is the duality between point space and parameter space:
- Each point in data space corresponds to a curve in parameter space
- Each model in parameter space corresponds to a point in data space
- Points lying on the same model generate curves that intersect at the model's parameters

#pagebreak()

=== Line Detection via Hough Transform <hough_lines>
For line detection, we parameterize lines using the normal form:
$ rho = x cos theta + y sin theta $ <line_normal_form>

where $rho$ is the perpendicular distance from the origin to the line, and $theta$ is the angle of the normal vector.

#algorithm-figure("Hough Transform for Line Detection", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([Hough-Transform], [edge points ${(x_i, y_i)}_(i=1)^N$], {
    Comment[*Input:* Edge points ${(x_i, y_i)}_(i=1)^N$]
    Comment[*Output:* Line parameters ${(rho_k, theta_k)}$]
    LineBreak
    Comment[Initialize accumulator array $A[rho, theta]$ with appropriate discretization]
    Assign[$A[rho, theta]$][$0$ for all $rho, theta$]
    LineBreak
    Comment[For each edge point $(x_i, y_i)$:]
    For($i = 1, dots, N$, {
      Comment[(a) For $theta = -pi/2$ to $pi/2$ with step $Delta theta$:]
      For($theta = -pi/2, dots, pi/2, "Step:" Delta theta$, {
        Comment[(i) Compute $rho = x_i cos theta + y_i sin theta$]
        Assign[$rho$][$x_i cos theta + y_i sin theta$]
        LineBreak
        Comment[(ii) Increment $A[rho, theta]$]
        Assign[$A[rho, theta]$][$A[rho, theta] + 1$]
      })
    })
    LineBreak
    Comment[Find local maxima in $A[rho, theta]$ above threshold]
    Assign[$"Peaks"$][$"FindLocalMaxima"(A, "threshold")$]
    LineBreak
    Comment[Each maximum corresponds to a line with parameters $(rho, theta)$]
    For([each peak $(rho_k, theta_k)$ in Peaks], {
      Comment[Line detected with parameters $(rho_k, theta_k)$]
    })
    LineBreak
    Return["Peaks"]
  })
})

*Advantages of Normal Parameterization:*
+ _Bounded Parameter Space_: $rho in [0, rho_"max"]$ and $theta in [-pi/2, pi/2]$
+ _No Singularities_: Unlike slope-intercept form, vertical lines are handled naturally
+ _Uniform Discretization_: Parameter space can be uniformly discretized

#pagebreak()

=== Extension to Circle Detection <hough_circles>
Circle detection requires a three-dimensional parameter space $(x_c, y_c, r)$ where $(x_c, y_c)$ is the center and $r$ is the radius. The circle equation is:
$ (x - x_c)^2 + (y - y_c)^2 = r^2 $ <circle_equation>

#algorithm-figure("Hough Transform for Circle Detection", vstroke: .5pt + luma(200), {
  import algorithmic: *
  Procedure([Hough-Transform-Circle], ([edge points ${(x_i, y_i)}_(i=1)^N$], [known radius $r$]), {
    Comment[*Input:* Edge points ${(x_i, y_i)}_(i=1)^N$, known radius $r$]
    Comment[*Output:* Circle centers ${(x_(c,k), y_(c,k))}$]
    LineBreak
    Comment[Initialize 2D accumulator $A[x_c, y_c]$]
    Assign[$A[x_c, y_c]$][$0$ for all $x_c, y_c$]
    LineBreak
    Comment[For each edge point $(x_i, y_i)$:]
    For($i = 1, dots, N$, {
      Comment[(a) For $theta = 0$ to $2pi$ with step $Delta theta$:]
      For($theta = 0, dots, 2pi$, {
        Comment[(i) Compute center coordinates]
        Assign[$x_c$][$x_i + r cos theta$]
        Assign[$y_c$][$y_i + r sin theta$]
        LineBreak
        Comment[(ii) Increment $A[x_c, y_c]$]
        Assign[$A[x_c, y_c]$][$A[x_c, y_c] + 1$]
      })
    })
    LineBreak
    Comment[Find peaks in $A[x_c, y_c]$]
    Assign[$"Peaks"$][$"FindLocalMaxima"(A, "threshold")$]
    LineBreak
    Return["Peaks"]
  })
})

For unknown radius, a 3D accumulator is required, significantly increasing computational cost.

#pagebreak()
