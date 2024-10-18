function x=randexp(n,m,lambda)
% randexp:    g�n�rateur de nombres al�atoires suivant une distribution
% exponentielle
%   USAGE: x=randexp(n,m,lambda);
%   lambda: scalaire ou tableau r�el positif donnant la valeur moyenne de la distribution
%  
%   p:   tableau d'entiers positifs ou nuls de taille [n m],
%      
%        Nota: Il faut n�cessairement que size(lambda)==[n m ].
%  
s=size(lambda);
  if ~(isequal([n,m],s) | isequal(s,[1 1]))
         error('L''argument "lambda" doit �tre un scalaire ou �tre de la taille du r�sultat demand�')
     end
     
unif=rand(n,m);

x=-lambda.*log(unif);