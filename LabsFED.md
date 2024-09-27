![](https://cdn.mathpix.com/cropped/2024_09_27_5764cda6987cdb496128g-01.jpg?height=109&width=229&top_left_y=228&top_left_x=702)

# FED 

## Fundamentals of estimation and detection in signals and images

## Labwork

1 Random variables (Sept., 16th) ..... 1
2 Parameter estimation in signal processing (Sept., 23th) ..... 7
3 Estimation for PALM microscopy, (Sept., 30th) ..... 11
4 Matched filter. Image registration (Oct., 7th) ..... 15
5 Detection using matched filter (Oct., 14th) ..... 19
Project: PALM Microscopy ..... 25

## Evaluation of the labworks

|  | Type of evaluation |
| :--- | :--- |
| Labwork 1 | No evaluation |
| Labwork 2 | You have to hand in a report |
| Labwork 3 | You have to use the results obtained during the labwork <br> for the project |
| Labwork 4 | You have to hand in a report |
| Labwork 5 | You have to hand in a multimedia file showing your work <br> on the last part of the labwork, with a short note explain- <br> ing your method (2 pages maximum). |

## Instructions for labwork reports

## General instructions

Take care of formal presentation: Presenting clearly your work is as important as doing it !

- Write sentences to explain your result. Copy of computer output is not acceptable !
- Use precise notation (use the notation introduced in the lectures...). For example : define an estimator $\widehat{\alpha}$ of parameter $\alpha$. Estimate its mean $\mathbf{E}[\widehat{\alpha}]$ and its variance $\operatorname{VAR}[\widehat{\alpha}]$.
- Give all information necessary to understand your experimental results. The result of a Monte Carlo simulation must come with the signal parameters and the number of realizations!
- The graphs must be cited and commented in the text, the used parameters must be mentioned.
- Do not include the code listing in your report

In conclusion : do not try to do everything very fast but understand what you do and prove that you have understood by writing a clear report.

## Format and deadlines

The reports should be written as two-column scientific articles of 6 pages max in pdf format. Examples and templates available on demand. The reports are to be handed in on Thursday evening the week after (before the beginning of the next lecture). Send your documents to: matthieu.boffety@institutoptique.fr.

## Labwork 1

## Random variables

## Contents

1 Histograms and correlation 1
1.1 Analysis of 1D signals . . . . . . . . . . . . . . . . . . . 1
1.2 Study of 2D data . . . . . . . . . . . . . . . . . . . . . . 2
2 First examples. Variance of an estimator. . . . . . . . . 3
2.1 Estimator of the mean of a Gaussian sample . . . . . . . 3
2.2 Estimator of the mean of an exponential sample . . . . . 4
2.3 Estimation of the location parameter of a Cauchy random variable

## 1 Histograms and correlation

### 1.1 Analysis of 1D signals

We first consider two random variables $X_{1}$ and $X_{2}$. We have at our disposal $N$ samples of each variable where $N=1000$. This samples are saved in the file RandomSamples.mat as two vectors of length 1000, denoted R_X1 and R_X2 respectively.
$\rightsquigarrow$ Using the command load RandomSamples, load the two vectors containing the $N$ samples of each variable and plot them.
$\rightsquigarrow$ Give (empirically) an estimation of the probability density function of the random variables $X_{1}$ and $X_{2}$.

Q1 Can we consider that the variables are both Gaussian? That their mean and variance are equal?

Q2 Can the $N$ samples of $X_{1}$ be considered as a realization of a white noise? Same question for $X_{2}$ ?

## Useful commands/tips:

To compute the histogram of a vector: [a,b]=hist(R_Xg, nbins); computes the histogram widespread on nbins intervals. Vector a contains the number of pixels in each interval and $b$ the values of the classes centers.

### 1.2 Study of 2D data

Using the command load NoisyImage, we now have at our disposal an image formed of a simple pattern distorted by noise.

Q3 Without any calculation, can we say that all regions of the image have the same probability density function and correlation function? Justify your answer qualitatively.
$\rightsquigarrow$ Numerically estimate the probability density function and correlation function of each region of the image.

Q4 Do these results confirm your first analysis?
Q5 From your observations in sections 1.1 and 1.2, can you comment on the relationship between the probability density function of a noise and its spectral characteristics?

## Useful commands/tips:

- subplot(311),plot(A), subplot(312), plot(B), subplot(313), plot(C): displays 3 plots vertically on the same figure.
- To handle the images:
- imshow, imagesc: to display images (in 255 gray levels, without or with scaling);
- colorbar: displays the color table;
- colormap('gray'): displays in gray levels;
- double: to convert a gray scale image to a double-precision coded image;
- VisuIdB, (toolbox SupOptique): to display images with a logarithmic scale;
- imcrop: to extract a region of the current image.
- To compute the histogram of a matrix, two ways:
- transform the matrix into a vector by the reshape command or the (:) syntax and then apply the previous hist command (recommended).
- use the imhist command.


## 2 First examples. Variance of an estimator.

### 2.1 Estimator of the mean of a Gaussian sample

Let us consider now a sample of $N$ independant gaussian elements. We assume that we know its variance $\sigma^{2}$. We wish to estimate its statistical mean (expectation) $m$ by the empirical mean $\widehat{m}$.
$\rightsquigarrow$ Generate such a sample with $N=100$. The parameters will be chosen as: $m=3$ and $\sigma=2$.
$\rightsquigarrow$ What is the result given by the empirical mean with your sample? Is it exactly equal too $m=3$ ?
$\rightsquigarrow$ Test this estimator a few times.

Q6 Calculate the bias and the variance of this estimator.
$\rightsquigarrow$ You shall now carry out a Monte-Carlo simulation to verify these results. Generate $K$ outcomes of the sample, then apply the expectation estimator on each of these realizations, and use the results to evaluate the variance of the estimator. You will take for instance $K=1000$ realizations .
$\rightsquigarrow \quad$ Test the estimator for different values of $N$.

Q7 Is the Monte-Carlo simulation consistent with the calculated values of the bias and the variance of the estimator?

## Useful commands/tips:

- $R_{-} X g=r a n d n(N, M)$; : to have a matrix $R_{-} X g$ filled with $N \times M$ independant realizations of a zero-mean Gaussian random variable of variance equal to one.
- mean(A), $\operatorname{var}(\mathrm{A})$ : empirical mean and empirical variance of vector A. When A is a matrix, mean ( A ) is a row vector containing the mean value of each column.


### 2.2 Estimator of the mean of an exponential sample

Let us now consider a sample made of $N$ independant elements from an exponential random variable. We wish to estimate its statistical mean $m$.

Q8 Propose an estimator. What are the bias and the variance of this estimator?
$\rightsquigarrow$ Test this result with a Monte-Carlo simulation.
$\rightsquigarrow$ Plot the result of an estimation for $m \in[0,1,2,3,4,5]$ with error bar corresponding to the standard deviation of the estimator.

Q9 Compare with the same curve for the Gaussian case. What are your comments?

## Useful commands/tips:

- R_Xexp=randexp( $m, n$, lambda ): creates a matrix made of $m \times n$ samples of an exponential random variable, the statistical mean of which is lambda.
- errorbar(X,Y,L,U) plots the graph of vector X vs. vector Y with error bars specified by the vectors $L$ and $U$.


### 2.3 Estimation of the location parameter of a Cauchy random variable

We now consider a sample made of $N$ elements from a Cauchy random variable $X$, its probability density function is given by

$$
p_{X}(x)=\frac{1}{\pi} \cdot \frac{1}{1+(x-a)^{2}}
$$

We wish to estimate the location parameter $a$.
Q10 What are the mean, the variance and the median of this random variable?
$\rightsquigarrow$ Test by Monte-Carlo simulations the empirical mean and the empirical median functions to estimate $a$.

## Useful commands/tips:

- R_Xcau=randcauchy $(m, n, a)$ : creates a matrix made of $m \times n$ samples of a Cauchy random variable of paramater $a$.
- median(X) For vectors, median( X ) is the median value of the elements in X . For matrices, median( X ) is a row vector containing the median value of each column.

LABWORK 1 : Random variables (Sept., 16th)

## Labwork 2

## Parameter estimation in signal processing

## Questions P1 to P4 must be prepared in advance

## Contents

1 Estimation under additive normal noise . . . . . . . . . 7
2 Exponential case ..... 9
3 Estimator comparison ..... 10

We consider the relaxation signal of a physical process :

$$
\begin{equation*}
s_{i}=s_{0} \exp ^{-\alpha_{0} i}, i \in[0, N-1] \tag{2.1}
\end{equation*}
$$

The objective here is to estimate $\alpha_{0}$ from a noisy signal. We consider successively two types of noises: normal (Gaussian additive) and exponential (multiplicative) noises.

## 1 Estimation under additive normal noise

We first assume that the experiment is perturbed by an additive white normal noise.

P1 Give the expression of the log-likelihood function $\ell(\alpha)$, then find the expression of the maximum likelihood estimator of $\alpha_{0}$. Can we give an analytical form?

P2 Give the expression of the Cramer-Rao Lower Bound (CRLB) for the estimation of $\alpha_{0}$. What does the ratio $s_{0} / \sigma$ represent physically? In some cases (which ones?) the expression of the CRLB can be simplified using:

$$
\sum_{n=0}^{+\infty} n^{2} q^{n}=\frac{q(1+q)}{(1-q)^{3}}, q<1
$$

$\rightsquigarrow$ Design a matlab function which gives a realization (length $N=100$ ) of the noisy signal. The arguments will be $s_{0}, \alpha_{0}$ and $\sigma$, the standard deviation of noise. You can choose the parameters as $\alpha_{0}=0.1, s_{0} / \sigma=10$.
$\rightsquigarrow$ Design a function which calculate the log-likelihood function $\ell(\alpha)$ using the previous realization. $\alpha$ should be an input parameter of the function.
$\rightsquigarrow \operatorname{Plot} \ell(\alpha)$ for $\alpha \in\left[\begin{array}{ll}0 & 1\end{array}\right]$.
$\rightsquigarrow$ Plot again this curve using other realizations.

Q1 What are your comments?
$\rightsquigarrow$ Design a function which finds the maximum likelihood estimator from a realization. You can use the matlab functions fzero or fminsearch.
$\rightsquigarrow \quad$ Estimate the statistical mean and variance of the estimator with a Monte Carlo simulation.

Q2 Is the estimator biased? Does it reach the CRLB?

## Useful commands/tips:

- fzero: tries to find a zero of a function. The syntax to use is: fzero( $@(x)$ fonc (x, a_1, a_2), x0) to find the zero of the function fonc ( $\alpha$, a_1 , a_2) according to $\alpha$. The a_1, a_2 parameter stay constant during optimization. XO is the initial value.
- fminsearch: attempts to find a local minimizer of a function. The syntax to use is: fminsearch(@(x) fonc(x, a_1, a_2), X0);


## 2 Exponential case

We now consider that the signal (2.1) is pertubed by exponential noise. Remember that for such process (with mean $I$ ) the probability density is :

$$
P(x)=\frac{1}{I} \exp \left[-\frac{x}{I}\right]
$$

P3 Express the maximum likelihood estimator of $\alpha_{0}$. Is it the same as the Gaussian case?

P4 Express the CRLB for $\alpha_{0}$ estimation. In order to obtain an analytical form, we can use the result:

$$
\sum_{n=0}^{N} n^{2}=\frac{N(N+1)(2 N+1)}{6}
$$

What are the parameters the CRLB depends on?
$\rightsquigarrow$ Design a Matlab function which gives a realization of the stochastic process.

Q3 On which parameters does it depend?
$\rightsquigarrow$ Design a function which finds the maximum likelihood estimator from an experiment. We can use the Matlab functions fzero or fminsearch.
$\rightsquigarrow$ Estimate statistical mean and variance of the estimator with a Monte Carlo simulation.

Q4 Is the estimator biased? Does it reach the CRLB?

## Useful commands/tips:

$\mathrm{R}=\mathrm{randexp}(1, \mathrm{n}, \mathrm{lambda})$ returns pseudo-random values drawn from an exponential distribution with means stored in the vector $(1 \times n)$ lambda.
10 LABWORK 2: Parameter estimation in signal processing (Sept., 23th)

## 3 Estimator comparison

$\rightsquigarrow \quad$ Apply the maximum likelihood estimator adapted to the normal noise to signal pertubed by exponential noise. Plot the estimate for several realizations. Estimate the variance.

Q5 What are your comments?

## Labwork 3

## Estimation for PALM microscopy

## Questions P1 to P4 must be prepared in advance

## Contents

1 Location estimation . . . . . . . . . . . . . . . . . . . . . 12
2 Nuisance parameters . . . . . . . . . . . . . . . . . . . . . 13
3 Influence of the imaging system characteristics . . . . . 14

Due to light diffraction, the resolution of conventional microscopic images is limited by the wavelength. Nevertheless various methods have been developed in order to "beat" the diffraction limit, especially in fluorescence microscopy. One of these methods is the photoactivated localization microscopy (PALM). This technique consist in isolating the observed fluorescent emitters and fitting their PSF. This way one can determine the location of the isolated emitter with a precision better than the Rayleigh criterion, achieving "super resolution". In this labwork, it is proposed to estimate the location of a given emitter.

For simplicity reasons and without any loss of generality, we consider the location from a 1D point of view. Let $x$ be the spatial coordinate, $\theta$ the position of an emitter. It is then assumed that the signal given by this single emitter on the image plan can be written as $s(x, \theta)=a r(x, \theta)+b$ where $r$ is a Gaussian function centered on $\theta$ so that

$$
r(x, \theta)=\frac{1}{\sqrt{2 \pi} \sigma_{r}} \exp \left[-\frac{(x-\theta)^{2}}{2 \sigma_{r}^{2}}\right]
$$

where $\sigma_{r}$ is expressed as a function of the full width at half maximum (FWHM) $w$ of the signal on the sensor. In this labwork, $b$ is assumed to be an additive normal white noise of standard deviation $\sigma_{b}$.

P1 We define $r_{i}$ the integration of $r(x, \theta)$ over the pixel $i$ with $i \in[0, N-1]$

$$
r_{i}=\int_{i \Delta x}^{(i+1) \Delta x} r(x, \theta) \mathrm{d} x
$$

where $\Delta x$ is the discretization step. Give the expression of $r_{i}$ as a function of the parameters $\theta$ and $w$. We remind that

$$
\begin{equation*}
\frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-u^{2}} \mathrm{~d} u=\operatorname{erf}(z) \tag{3.1}
\end{equation*}
$$

## 1 Location estimation

We assume in this section that $a$ is known and $a=1$.

P2 Give the expressions of the CRLB and the Maximum Likelihood (ML) estimator for the estimation of $\theta$.
$\rightsquigarrow$ Assuming $N=100, w=2 \mu \mathrm{~m}, \Delta x=2 \mu \mathrm{~m}$, plot the the signal $r\left(x, \theta_{0}\right)$ versus $x$ for a few values of $\theta$ (for example $\theta=\theta_{0} \in\{46.047 .248 .449 .650 .852 .0\}$ (in $\mu \mathrm{m}$ ).
$\rightsquigarrow$ On the corresponding figures, plot simultaneously the signal $r\left(x, \theta_{0}\right)$ and $r_{i}$ for a few values of $\theta$ (for example $\theta=\theta_{0} \in\{46.047 .248 .449 .650 .852 .0\}$ (in $\mu \mathrm{m}$ ). You will plot each curve on a different figure using for example subplot and bar Matlab functions. For these figures, you can reduce the $x$-interval to $[40 \mu \mathrm{~m}, 60 \mu \mathrm{~m}]$.
$\rightsquigarrow$ On another figure, plot the CRLB for the estimation of $\theta$ as a function of $\theta$. You can choose $\sigma_{b}=0.01$.

Q1 Comment the shape of the CRLB curve, its dependence to $\theta$. Considering the behavior of $r_{i}$ with $\theta$, propose an interpretation about the shape of the CRLB.
$\rightsquigarrow$ Design a Matlab script or function to generate a realization of the noisy signal.
$\rightsquigarrow$ Design a Matlab script or function which finds the maximum likelihood estimator from a realization. You can use the Matlab functions fzero or fminsearch.
$\rightsquigarrow$ Estimate the statistical mean and variance of the ML estimator with a Monte-Carlo simulation.

Q2 Can we say that the estimator is efficient? Justify your answer.

## Useful commands/tips:

- $\operatorname{bar}(\mathrm{X}, \mathrm{Y})$ : draws the columns of the M-by-N matrix Y as M groups of N vertical bars. The vector X must not have duplicate values.
- $\operatorname{subplot}(\mathrm{n}, \mathrm{m}, \mathrm{k})$ : divides the current figure window in $n \times m$ subfigure and considers the $k^{\text {th }}$ subfigure. The subfigures are numbered from left to right and from the top to the bottom.
- erf: is the error function defined by equation (3.1)
- hold on: holds the current plot and all axis properties so that subsequent graphing commands add to the existing graph.
- fzero: tries to find a zero of a function. The syntax to use is : fzero(@(x) fonc(x,a_1, a_2), X0) to find the zero of the function fonc( $\alpha$, a_1 , a_2) according to $\alpha$. The a_1, a_2 parameter stay constant during optimization. XO is the initial value.
- fminsearch: attempts to find a local minimizer of a function. The syntax to use is : fminsearch(@(x) fonc(x, a_1, a_2), X0);


## 2 Nuisance parameters

We now assume that $a$ is unknown, since it is of no interest for the location of the emitter it is considered as a nuisance parameter.

P3 Propose a method to deal with this kind of parameters.
P4 Give the expressions of the CRLB and the ML estimator for the estimation of $\theta$ in this case.
$\rightsquigarrow$ Plot the CRLB of the estimation of $\theta$ as a function of $\theta$.

Q3 Compare it with the case where $a$ was known: what can you tell? Was this result expected and why?
$\rightsquigarrow$ Design a Matlab script or function which finds the maximum likelihood estimator from a realization in this case.
$\rightsquigarrow$ Estimate the statistical mean and variance of the ML estimator with a Monte-Carlo simulation.

Q4 What are your comments?

## 3 Influence of the imaging system characteristics

We still assume that $a$ is unknown. The width $w$ can be seen as a parameter that depends on the imaging system characteristics, as it can be tuned by changing the magnification of the system. We propose here to find the value of $w$ that will lead to the best location estimation.
$\rightsquigarrow \quad$ Plot the CRLB calculated in section 2 for various values of $w$ (for example $w \in\left\{\begin{array}{llll}1 & 2 & 3 & 4\end{array}\right\}$ ).

Q5 Comment the shape of the curves. How does the CRLB evolve according to $w$ ?
$\rightsquigarrow \quad$ Plot the evolution of maximum of the CRLB as a function of $w$.
Q6 Comment the curve. How would you choose the magnification of your imaging system to have the best location precision? What is the expected resolution in this case?

## Labwork 4

## Estimation using the matched filter. Image registration.

## Questions P1 and P2 must be prepared in advance

## Contents

1 Global registration . . . . . . . . . . . . . . . . . . . . . . 15
2 Registration by using a region of the image . . . . . . . 16

We consider an image sequence $f_{k}(x, y), k \in[1, K], x \in[1, M], y \in[1, N]$, each image of the sequence being translated of $\left(x_{k}, y_{k}\right)$ relative to the first one, where $\left(x_{k}, y_{k}\right)$ is only known to be an integer number of pixels. Our goal is to register these images. We assume that these images are perturbed by $b_{k}(x, y)$, a centered Gaussian additive noise of variance equal to $\sigma^{2}$. The signal model is thus:

$$
\forall k>1, f_{k}(x, y)=f_{1}\left(x-x_{k}, y-y_{k}\right)+b_{k}(x, y)
$$

and we want to estimate all the parameters $\left(x_{k}, y_{k}\right)$.

## 1 Global registration

P1 Is the solution given by the matched filter optimal here?
P2 Propose then a registration algorithm for this sequence.
$\rightsquigarrow \quad$ Apply it on the sequence contained in the file ShiftedImages .dat. The first image of the series will be used as the reference image.
$\rightsquigarrow$ You shall construct a sequence with the registered images to check that they are well superposed.
$\rightsquigarrow$ You shall also plot, on the same figure, the values of the estimated translations $\hat{x}_{k}$ and $\hat{y}_{k}$, as a function of $k$.

## Useful commands/tips:

- $[I, J]=$ find $(X>5)$ returns the row and column indices of the matrix X where X is greater than 5 .
- fft2 2D Fourier transform of an image.
- ifft2 2D inverse Fourier transform of an image.
- fftshift circular shift of an image so that the left top point becomes the central point.
- To successively display images of a sequence: getframe, movie. Example :
figure
for $i=1: 6$
imagesc(Images(:,:,i)), colormap(gray);
$\operatorname{Mp}(\mathrm{i})=$ getframe(gcf);
end;

![](https://cdn.mathpix.com/cropped/2024_09_27_5764cda6987cdb496128g-22.jpg?height=63&width=1166&top_left_y=1736&top_left_x=316)
where the values of the (3-dimensional) array Images vary between 0 and 1 .

## 2 Registration by using a region of the image

For sake of memory resources or because of time calculation, it is not always possible do compute the Fourier transform of the whole image. In this case, one has to compute the shift from a small region of the reference image. You will take a $30 \times 30$ pixels wide region.

The goal is then to choose the region of the image that leads to the highest precision for shift estimation.
$\rightsquigarrow \quad$ Choose 5 very distinct small regions in the reference image.
$\rightsquigarrow$ Carry out the previous algorithm using these regions, and compare the results.
$\rightsquigarrow \quad$ What kind of image features lead to accurate shift estimation?
Q1 Which criterion could have been used to foresee this result?
$\rightsquigarrow$ Using this criterion, find over the whole image the best $30 \times 30$ pixels wide region.

## Useful commands/tips:

- M=SpectralWidth2(Im); estimate the square of the "spectral width" of a digital image $\operatorname{Im}$ using fft2 algorithm. M is $2 \times 2$ matrix.

The square of the "spectral width" of an image $A(x, y)$ is defined by the matrix :

$$
\left(\begin{array}{cc}
\Delta_{\text {horiz }}^{2} & \Delta_{\text {diag }}^{2} \\
\Delta_{\text {diag }}^{2} & \Delta_{\text {vert }}^{2}
\end{array}\right)=\left(\begin{array}{ll}
\int \sigma^{2}|\widetilde{A}(\sigma, \mu)|^{2} d \sigma d \mu & \int \sigma \mu|\widetilde{A}(\sigma, \mu)|^{2} d \sigma d \mu \\
\int \sigma \mu|\widetilde{A}(\sigma, \mu)|^{2} d \sigma d \mu & \int \mu^{2}|\widetilde{A}(\sigma, \mu)|^{2} d \sigma d \mu
\end{array}\right)
$$

where $\widetilde{A}(\sigma, \mu)$ is the Fourier transform of $A(x, y)$.

## Labwork 5

## Detection using matched filter

## Contents

1 Characterizing the detector performance . . . . . . . . 20
2 Object detection in a real-time video . . . . . . . . . . . 22
2.1 Matched Filter on images . . . . . . . . . . . . . . . . . 22
2.2 Real-time Video . . . . . . . . . . . . . . . . . . . . . . . 23

In radar detection, one must test two hypotheses :
Hypothesis H0 $\quad x(t)=b(t)$
Hypothesis H1 $\quad x(t)=a s(t)+b(t)$
where :

- $x(t)$ is the measured signal,
- $s(t)$ is the emitted pulse, known by the user,
- $a$ is the attenuation factor,
- $b(t)$ is the noise on the measure.

We assume that we know where the target may be. We only want to know if the target is or not at this position.

We first only consider the detection problem. In reality, we don't know if there is a target, neither its possible position. We must detect and estimate at the same time. That is what we will study in the image processing application.

If the noise $b(t)$ is white, the optimal detector (matched filter) is a correlation between the measured signal and the emitted pulse :

$$
F(\theta)=\int x(t) s(t-\theta) d t
$$

We only need to calculate this expression at one point (chosen here as 0 ) if we assume that the position is known. Thus, we have to test :

$$
F(0)=\int x(t) s(t) d t
$$

This value is then compared to a threshold in order to take a decision.

![](https://cdn.mathpix.com/cropped/2024_09_27_5764cda6987cdb496128g-26.jpg?height=235&width=1235&top_left_y=1002&top_left_x=365)

## 1 Characterizing the detector performance

The SNR (Signal to Noise Ratio) of the detection is defined by :

$$
\mathrm{SNR}=\frac{a^{2} E_{s}}{\sigma^{2}} \quad \mathrm{SNR}_{\mathrm{dB}}=10 \log \left(\frac{a^{2} E_{s}}{\sigma^{2}}\right)
$$

where $E_{s}$ is the energy of the emitted pulse: $E_{s}=\int|s(t)|^{2} d t$ and $\sigma^{2}$ the variance of the noise $b(t)$.

We assume that the temporal shape of the pulse is gaussian and the attenuation factor $a=1$. We suppose also that $b(t)$ is a gaussian white noise.
$\rightsquigarrow$ Design a Matlab function which gives a realization of the noisy signal $x$. Length of the vector will be chosen as 255 , duration of the pulse as 40 samples and SNR as 12 dB .
$\rightsquigarrow$ Plot the noisy pulse and the noise alone (hypothesis 0 and 1 ). Apply the matched filter to these two signals.

Q1 Compare the two results. What threshold value do you choose?
$\rightsquigarrow$ Run your code several times.

Q2 What do you think of the threshold value first chosen ?

## Useful commands/tips:

- gausswin( $n, n /$ duration) gaussian pulse in a vector of length $n$. The duration is defined at $1 / \sqrt{\mathrm{e}} \approx 0.6$.
- $\operatorname{sum}(\mathrm{s} . \wedge 2$ ) to compute the energy of a vector s
- Calculating the cross-correlation at 0 can be easily done by a scalar product of two line vectors as $\mathrm{c}=1 / \mathrm{N} * \mathrm{x} * \mathrm{~s}^{\prime}$;
$\rightsquigarrow$ In order to carry out a Monte Carlo simulation, design two matrices (under H0 and H1) of 1000 realizations of the signal.


## Useful commands/tips:

- ones $(1000,1) * \operatorname{s}+r a n d n(1000, n)$ where $s$ is a line vector of dimension n corresponding to the emitted pulse, give a matrix $1000 \times n$. Each line of this matrix is a realization of the noisy signal.
$\rightsquigarrow$ Apply the matched filter to these 1000 realizations.


## Useful commands/tips:

The cross-correlation at 0 for all lines of a matrix : $c=1 / \mathrm{N} * \mathrm{X} * \mathrm{~s}$ '; (s line vector, X matrix, C column vector)
$\rightsquigarrow \quad$ Plot the histograms of the results under the two hypothesis H 0 and H 1 .

Q3 What is the influence of the threshold value on the detection probability (true positives) and on the false alarm probability (false positives)?

## Useful commands/tips:

To plot two histograms on the same graphic :
$[\mathrm{N} 1, \mathrm{X} 1]=$ hist $(c 1,20) ;[\mathrm{NO}, \mathrm{XO}]=$ hist $(\mathrm{co}, 20)$;
plot(X1, N1, XO, NO), legend('H1', 'HO')
$\rightsquigarrow \quad$ The ROC (Receiver Operating Characteristic) is a curve of the detection probability versus false alarm probability for all the threshold possible values. Plot the ROC curve from your Monte Carlo simulation.
$\rightsquigarrow$ Plot again the histograms and the ROC curve for simulation with SNR of 6 and 20 dB .
$\rightsquigarrow$ Plot now the ROC curves for SNR varying from 0 to 20 dB .

Q4 What is the limit of the ROC curve when SNR decreases? Why?

## 2 Object detection in a real-time video

The aim of this part is to detect an objet on a real time video from the webcam. First of all, you shall test your algorithm on good quality images.

### 2.1 Matched Filter on images

Here we wish to detect a character in a text. In this case we don't know its position. We must address the "detection-estimation" problem. The matched filter can be used for that purpose :

- a peak in the matched filter output indicate if the character is detected,
- the peak position gives an estimate of the character position.
$\rightsquigarrow$ Design and test such an algorithm on good quality images (parfait.bmp for instance). The character to detect will be selected on the image.


## Useful commands/tips:

- imread read image from graphics file, example:
a=imread('parfait.bmp').
- double convert to double precision, example ad=double(a);
- $a=m e a n(m e a n(a))-a$; by this command a white background image $a$ is transformed in a zero mean, black background one. Use this command before computing the matched filter.
- imcrop crops an image to a specified rectangle.
- In order to seek peak positions and display a red 'x-mark' on each :
$[\mathrm{I}, \mathrm{J}]=\mathrm{find}(\operatorname{cs} 2>0)$;
figure, imshow(text), hold on, plot(J,I,'xr');


### 2.2 Real-time Video

$\rightsquigarrow$ Set the webcam resolution to 640x480pixels, using the Logitech software.
$\rightsquigarrow$ Execute the program WebcamImageCapture and take a look at the program file by using the command edit WebcamImageCapture.m.

Q5 What is the name, the size and the format of the acquired image?
$\rightsquigarrow \quad$ By copying the source code of the program, design a program which can detect an track an object (of your choice) in the webcam image.

## Useful commands/tips:

- rgb2gray convert a color image on a (8 bits) gray-level image.
- WebcamImageCapture.m is a script which acquire and display images from the webcam using the Image Acquisition Toolbox.


## PALM Microscopy

## Outline of the project

PALM microscopy consists in estimating the position of fluorophores with subpixel resolution in order to reconstruct images having a resolution better than the diffraction limit.

The goal of the project is

- To study the resolution that can be reached by this method, and the optimal parameters of the imaging system parameter. To simplify, this study is to be done on 1 dimensional signal.
- To implement the algorithm for reconstruction a higher resolution image from a sequence of images of isolated fluorophores, and apply it to a test image.

This work is to be reported under the form of a scientific journal paper with an archive file containing your source codes. To this end, templates of the journal "Optics Express" are posted on Libres Savoirs (in MS Word and Latex), together with an example of Optics Express paper. The length of the paper is limited to 10 pages.

The features that must be reported in this work are:

- Precision analysis (1D):
- Simple and quick analysis of the noise of the given images, analysis of the CRLB when the parameter $a$ of the PSF is unknown, especially its variation with $\theta$, according to the noise model (previously found).
- Expression of the ML estimator, and estimation of its bias and efficiency.
- Optimal value of the PSF width $w$.
- Implementation of the algorithm (2D)
- Algorithm for detection and rough estimation of the position of single fluorophores in an image.
- Expression and implementation of ML position estimator in 2D.
- Application of the method to the proposed images and find the hidden pattern.

For this project, the PSF is supposed to be an isotropic Gaussian pulse with the same FWHM $w=2$ in the two directions and maximal value $a=1$. The given numerical values are dimensionless.

The previous points are only guidelines. You are free to order them as you want. The only requirement is that the "flow" of the paper is logical.

The article is to be handed, in PDF format, on October 28th.

## Data (available on Libres Savoirs )

- ImageTest.mat: reference image for testing your algorithm.
- CoordinatesTest.mat: exact positions of the fluorophores in image ImageTest.mat.
- ImagesPALM.mat: sequence of 865 PALM images.
- BlurredImage.png: low resolution image of the pattern to reconstruct from ImagesPALM.mat sequence.


## Matthieu's tips

1. Try to use elementary functions as best as you can in your code (through the use of Matlab functions rather than long scripts or long lists of commands). Indeed small functions are easier to read and debug.
2. If you use Matlab scripts, do not forget to clear your variables from time to time to avoid issues such as unwanted changes in vector or matrix sizes.
3. In the case of an exhaustive, but coarse, search of multiple local maxima in an image, you can proceed as follows: search for the global maximum, then set the corresponding and surrounding pixels to zero, search for the next maximum, then set its pixel and the surrounding ones to zero and repeat the operation until you have found all the wanted maxima.
4. For a precise (subpixel) estimation of an extremum, try to use the Matlab function fminsearch which works fine in 2D...
5. This project implies to process a pretty large amount of images. Therefore, before naively using your code on all the images at once,

- First, try your algorithm on a single image for which you know the exact location of the fluorophores (for example by using the files ImageTest.mat and CoordinatesTest.mat which are available on Libres Savoirs)
- Then, estimate the computation time needed for the whole set of images by calculating it for one image (through the use of the tic toc functions for example): It can be interesting to know if you have to let your program run all night long for example...

6. Do not forget to correctly read the Matlab error messages and do not forget that the doc and help commands give you access to the help instructions of a given function/command.
7. Any questions? Any overwhelming issues? You know how to contact us.
