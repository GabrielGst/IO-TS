function PSFi = PSFdiscrete( sigs, Dx, theta, I )
% PSFdiscrete: calcule le signal discretise et integre du a un emetteur 
% unique
% USAGE: PSFi = PSFdiscrete(w,Dx,theta,I)
% ENTREES 
%   sigs  : ecart-type de la PSF
%   Dx    : pas de discretisation des mesures  
%   theta : position de l'emetteur ayant generer le signal
%   I     : indice des echantillons de mesures
% SORTIES
%   PSFi  : valeur de la PSF, dimension N=length(I)
%
% Auteur  : M. Boffety
% Version : M. Boffety - 6/10/2017
%


    % "bornes" du calcul du signal discretise
    Ai=1/sqrt(2)*(I*Dx-theta)/sigs;
    Bi=1/sqrt(2)*((I+1)*Dx-theta)/sigs;    
    
    % calcul de la PSF suivant l'expression trouvee a la question P1
    PSFi=0.5*(erf(Bi)-erf(Ai));

end

