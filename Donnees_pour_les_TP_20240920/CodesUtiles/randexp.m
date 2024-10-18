function x=randexp(n,m,lambda)
% randexp:    générateur de nombres aléatoires suivant une distribution
% exponentielle
%   USAGE: x=randexp(n,m,lambda);
%   lambda: scalaire ou tableau réel positif donnant la valeur moyenne de la distribution
%  
%   p:   tableau d'entiers positifs ou nuls de taille [n m],
%      
%        Nota: Il faut nécessairement que size(lambda)==[n m ].
%  
s=size(lambda);
  if ~(isequal([n,m],s) | isequal(s,[1 1]))
         error('L''argument "lambda" doit être un scalaire ou être de la taille du résultat demandé')
     end
     
unif=rand(n,m);

x=-lambda.*log(unif);