% ----------------------------------
%%% Paramètres
a           = 1;   % amplitude of the PSF
w           = 2;   % FWHM of the PSF [µm]
sigma_r     = w/(2*sqrt(2*log(2)));   % standard deviation of the PSF [µm]
Dx          = 2;   % Discretization step [µm]
theta0  = [46.0 47.2 48.4 49.6 50.8 52.0];   % test values of theta
N           = 100;   % number of pixels
I           = linspace(0,N-1,N);   % index of the  pixels
X           = linspace(I(1)*Dx,I(end)*Dx,N*10); % position to plot the "un-discretized" Gaussian
% ----------------------------------

%%

% ----------------------------------
%%% PSF profile
r = NaN(length(X),length(theta0));
r_i = NaN(length(I),length(theta0));

figure
for k=1:length(theta0)
   % 'Discretized' PSF 
   r(:,k) = Dx/(sqrt(2*pi)*sigma_r)*exp(-0.5*(X-theta0(k)).^2/sigma_r^2);
   % 'Integrated' PSF
   r_i(:,k)=PSFdiscrete(sigma_r,Dx,theta0(k),I);
   % display
   subplot(2,3,k), hold on
   bar((I+0.5)*Dx,r_i(:,k))
   % halp step shift to be in the center of the interval [i, i+1]
   plot(X,r(:,k),'r')
   xlabel('Position (µm)');
   ylabel('Signal (a.u.)');
   axis([41 58  0 max(r(:,k))])
   grid on
end
% ----------------------------------

