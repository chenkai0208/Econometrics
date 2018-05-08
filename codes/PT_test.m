% The Pesaran-Timmermann Test
% Pesaran and Timmermann (1992) present a non-parametric test 
% to examine the ability of a forecast to predict the direction of change 
% in a series of interest.
function Pvalue = PT_test(x,y)
len = length(x);
Px = 0;
Py = 0;
P = 0;
for i=1:len
    if x(i) > 0
        Px = Px+1;
    end
    if y(i) > 0
        Py = Py+1;
    end
    if x(i)*y(i) > 0
        P = P+1;
    end
end

P = P / len;
Px = Px / len;
Py = Py / len;

Pstar = Py*Px + (1-Py)*(1-Px);
Vp = Pstar*(1-Pstar)/len;
Vpstar = ((2*Py-1)^2)*(Px*(1-Px))/len + ((2*Px-1)^2)*(Py*(1-Py))/len + 4*Py*Px*(1-Py)*(1-Px)/(len*len);

PT = (P-Pstar) / sqrt(Vp-Vpstar);

Pvalue = 1-normcdf(PT);


