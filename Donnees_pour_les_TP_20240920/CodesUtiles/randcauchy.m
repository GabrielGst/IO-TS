function x=randcauchy(m,n,a)
% randcauchy:   générateur de nombres aléatoires suivant une distribution
%  de Cauchy de paramètre d'échelle égal à 1 et de paramètre de
%  localisation a.
%   USAGE: x=randcauchy(n,m,a);
%   a: scalaire ou tableau réel positif donnant le paramètre de la
%   localisation
%
%        Nota: Il faut nécessairement que size(a)==[m n].
% FB sept 11  
s=size(a);
  if ~(isequal([n,m],s) | isequal(s,[1 1]))
         error('L''argument "a" doit être un scalaire ou être de la taille du résultat demandé')
     end
     
x=(randn(m,n)./rand(m,n))+a;