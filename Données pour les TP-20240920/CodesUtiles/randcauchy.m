function x=randcauchy(m,n,a)
% randcauchy:   g�n�rateur de nombres al�atoires suivant une distribution
%  de Cauchy de param�tre d'�chelle �gal � 1 et de param�tre de
%  localisation a.
%   USAGE: x=randcauchy(n,m,a);
%   a: scalaire ou tableau r�el positif donnant le param�tre de la
%   localisation
%
%        Nota: Il faut n�cessairement que size(a)==[m n].
% FB sept 11  
s=size(a);
  if ~(isequal([n,m],s) | isequal(s,[1 1]))
         error('L''argument "a" doit �tre un scalaire ou �tre de la taille du r�sultat demand�')
     end
     
x=(randn(m,n)./rand(m,n))+a;