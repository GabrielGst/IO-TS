---
title: "Report of Estimation Practical"
author: "Gabriel Gostiaux"
date: "2024-10-18"
fontfamily: helvet
geometry: margin=2cm
linkcolor: PineGreen
output: 
    html_document:
        css: styles.css
        self_contained: no
    #header-includes: |
	pdf_document:
        #toc: true
        #number_sections: true
        #fig_width: 7
        #fig_height: 6
        #fig_caption: true
        #df_print: kable # print.data.frame(default)/kable/tibble
        #highlight: tango
        #latex_engine: xelatex # try to use xelatex as default
        #keep_tex: true
        #template: template.tex
        includes:
            #in_header: preamble.tex
            #before_body: doc-prefix.tex
            #after_body: doc-suffix.tex
---



## Labwork 2: Parameter Estimation in Signal Processing
### Gabriel Gostiaux

## Contents

1. Estimation under additive normal noise
2. Exponential case
3. Estimator comparison

---

We consider the relaxation signal of a physical process:

$$
s_{\text{clean}} = s_{0} e^{-\alpha_{0} i}, \quad i \in [0, N-1]
$$

The objective here is to estimate $\alpha_{0}$ from a noisy signal. We consider successively two types of noises: normal (Gaussian additive) and exponential (multiplicative) noises.

## 1. Estimation Under Additive Normal Noise

We first assume that the experiment is perturbed by an additive white normal noise.

- **P1** Give the expression of the log-likelihood function $\ell(\alpha)$, then find the expression of the maximum likelihood estimator of $\alpha_{0}$. Can we give an analytical form?

To compute the log-likelihood, we need to find an expression of the probability law of the noisy signal. Here, in the Gaussian additive noise, the resulting signal is simply a Gaussian signal centered on the true signal and with the variance of the noise.

$$
\begin{aligned}
P(x_i) &= \frac{1}{\sqrt{2\pi} \sigma} \times \exp\left(-\frac{(s_i - s_0 \times \exp(-\alpha_0 i))^2}{2\sigma^2}\right) \\
l_{\alpha} &= -\frac{N}{2} \times \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=0}^{N-1} \left[s_i - s_0 \times \exp(-\alpha i)\right]^2
\end{aligned}
$$

The MLE is then obtained for $\frac{\partial l_{\alpha}}{\partial \alpha} = 0$ and should attain the Cramer Rao Lower Bound (CRLB) defined by $\frac{\partial^2 l_{\alpha}}{\partial \alpha^2} = 0$:

$$
\frac{\partial l_{\alpha}}{\partial \alpha} = \frac{s_0}{\sigma^2} \sum_{i=0}^{N-1} \left[i \times \exp(-\alpha_0 i) \times \left(s_i - s_0 \times \exp(-\alpha_0 i)\right)\right] = 0
$$

- **P2** Give the expression of the Cramer-Rao Lower Bound (CRLB) for the estimation of $\alpha_{0}$. What does the ratio $\frac{s_{0}}{\sigma}$ represent physically?

$$
\frac{\partial^2 l_{\alpha}}{\partial^2 \alpha} = - \frac{s_0}{\sigma^2} \sum_{i=0}^{N-1} \left[i^2 \times \exp(-\alpha_0 i) \times \left(s_i - 2 \times s_0 \times \exp(-\alpha_0 i)\right)\right]
$$

In some cases (which ones?), the expression of the CRLB can be simplified using:

$$
\sum_{n=0}^{+\infty} n^{2} q^{n} = \frac{q(1+q)}{(1-q)^{3}}, \quad q < 1
$$

Assuming the mean of the measured signal is the clean signal, we can use the formula:

$$
\frac{\partial^2 l_{\alpha}}{\partial^2 \alpha} = -\frac{s_0^2}{\sigma^2} \sum_{i=0}^{+\infty} \left[i^2 \times \exp(-2\alpha_0 i)\right]
$$

The CRLB is then equal to:

$$
\text{CRLB} = \mathcal{E}\left(-1 / \frac{\partial^2 l_{\alpha}}{\partial^2 \alpha}\right) = \frac{\sigma^2}{s_0^2} \times \frac{(1-e^{-2\alpha_0})^3}{e^{-2\alpha_0}(1+e^{-2\alpha_0})}
$$

---

$\rightsquigarrow$ Design a MATLAB function which gives a realization (length $N=100$) of the noisy signal. The arguments will be $s_{0}, \alpha_{0}$ and $\sigma$, the standard deviation of noise. You can choose the parameters as $\alpha_{0} = 0.1$, $\frac{s_{0}}{\sigma} = 10$.

$\rightsquigarrow$ Design a function which calculates the log-likelihood function $\ell(\alpha)$ using the previous realization. $\alpha$ should be an input parameter of the function.

$\rightsquigarrow$ Plot $\ell(\alpha)$ for $\alpha \in \left[0, 1\right]$.

$\rightsquigarrow$ Plot again this curve using other realizations.

![&nbsp;](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_8_0.png)

![png](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_8_1.png)

![png](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_8_2.png)

Q1: What are your comments?

One can assume this function of $\alpha$ admits a maximum. On the figure, we can estimate $\alpha_0 = 0.1$.

---

$\rightsquigarrow$ Design a function which finds the maximum likelihood estimator from a realization. You can use the MATLAB functions `fzero` or `fminsearch`.

**Decay estimator that minimizes the function**: [0.10242188]

$\rightsquigarrow$ Estimate the statistical mean and variance of the estimator with a Monte Carlo simulation.

![png](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_14_0.png)

Q2: Is the estimator biased? Does it reach the CRLB?

The CRLB for a decay rate equal to 0.1 is $4.000028574288418 \times 10^{-5}$.

Since the variance equals the CRLB, the estimator is efficient, and it is not biased since it does not differ from the exact theoretical value as shown by the Monte Carlo simulation.

## 2. Exponential Case

We now consider that the signal (2.1) is perturbed by exponential noise. Remember that for such a process (with mean $I$) the probability density is:

$$
P(x) = \frac{1}{I} \exp \left[-\frac{x}{I}\right]
$$

- **P3** Express the maximum likelihood estimator of $\alpha_{0}$. Is it the same as the Gaussian case?

The resulting random variable follows an exponential probability law shifted by the deterministic signal:

$$
\begin{aligned}
P\left(s_i \mid \alpha\right) &= \frac{1}{s_{\text{clean}}} \exp \left(-\frac{s_i}{s_{\text{clean}}}\right) \\
l_{\alpha} &= -\sum \ln \left(s_{\text{clean}}\right) - \sum \left(\frac{s_i}{s_{\text{clean}}}\right) \\
\frac{\partial l_{\alpha}}{\partial \alpha} &= \frac{\partial}{\partial \alpha}\left[-\sum \ln \left(s_0 e^{-\alpha i}\right) - \sum \frac{s_i}{s_0} e^{\alpha i}\right] \\
\frac{\partial l_{\alpha}}{\partial \alpha} &= \frac{\partial}{\partial \alpha}\left[-N \ln (s_0) + \alpha \sum i - \sum \frac{s_i}{s_0} e^{\alpha i}\right]
\end{aligned}
$$

This results in:

$$
\begin{aligned}
\frac{\partial l_{\alpha}}{\partial \alpha} & = \sum i - \sum \frac{s_i}{s_0} i e^{\alpha i} \\
0 &= \sum i - \sum \frac{s_i}{s_0} i e^{\alpha_{0} i} \\
0 &= \sum i \left( 1 - \frac{s_i}{s_0} e^{\alpha_{0} i}\right)
\end{aligned}
$$

- **P4** Express the CRLB for $\alpha_{0}$ estimation. In order to obtain an analytical form, we can use the result:

$$
\begin{aligned}
\frac{\partial^2 l_{\alpha}}{\partial^2 \alpha} & = -\frac{1}{s_0} \sum s_i i^2 e^{\alpha i} \\
\left\langle \frac{\partial^2 l_{\alpha}}{\partial^2 \alpha}\right\rangle & = -\frac{1}{s_0} \sum \left\langle s_i \right\rangle i^2 e^{\alpha i} \\
& = -\sum i^2
\end{aligned}
$$

$$
\sum_{n=0}^{N} n^{2} = \frac{N(N+1)(2N+1)}{6}
$$

$$
\text{CRLB} = -\frac{1}{\left\langle \frac{\partial^2 L_{\alpha}}{\partial^2 \alpha}\right\rangle} = \frac{6}{N(N+1)(2N+1)}
$$

What are the parameters the CRLB depends on?

$\rightsquigarrow$ Design a MATLAB function which gives a realization of the stochastic process.

![png](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_21_0.png)

![png](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_21_1.png)

![png](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_21_2.png)
Q3: On which parameters does it depend?

$\rightsquigarrow$ Design a function which finds the maximum likelihood estimator from an experiment. We can use the MATLAB functions `fzero` or `fminsearch`.

**Decay estimator that minimizes the function**: [0.08886719]

$\rightsquigarrow$ Estimate the statistical mean and variance of the estimator with a Monte Carlo simulation.

![png](IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_files/IO24%20-%20TS%20-%20lab%202.2%20-%20estimation_27_0.png)

Q4: Is the estimator biased? Does it reach the CRLB?

Since the variance equals the CRLB, the estimator is efficient, and it is not biased since it does not differ from the exact theoretical value as shown by the Monte Carlo simulation.

## Useful Commands/Tips:

$\mathrm{R} = \mathrm{randexp}(1, n, \lambda)$ returns pseudo-random values drawn from an exponential distribution with means stored in the vector $(1 \times n)$ lambda.

## 3. Estimator Comparison

$\rightsquigarrow$ Apply the maximum likelihood estimator adapted to the normal noise to the signal perturbed by exponential noise. Plot the estimate for several realizations. Estimate the variance.

Q5: What are your comments?

-> to be continued
