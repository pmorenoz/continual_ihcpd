# Continual Learning for Infinite Hierarchical Change-Point Detection

We derive a continual learning mechanism that recursively infers the surrogate latent variable model that we plug in the Bayesian change-point detection (CPD) method. It is based on the sequential construction of the chinese restaurant process (CRP) and the expectation-maximization (EM) algorithm with stochastic gradient updates.

<img src="tmp/ihcpd.png" width="400"> <img src="tmp/illustration_threads.png" width="400">

## Potential Applications
Infinite Hierarchical CPD models can be used, for instance, to detect change-points in high-dimensional time-series where the number of likelihood parameters is much larger than the partitions (segments between change-points). 

## Contributors

[Pablo Moreno-Muñoz](http://www.tsc.uc3m.es/~pmoreno/), [David Ramírez](https://ramirezgd.github.io/) and [Antonio Artés-Rodríguez](http://www.tsc.uc3m.es/~antonio/)

For further information or contact:
```
pmoreno@tsc.uc3m.es
```
