
function z=Sphere(x)

  z=sum(x.^2);
%z =sum(rosenbrockfcn(x)) ;%2
%  z = sum(quarticfcn(x)) ;%3
%    z =sum(x+0.5).^2 ;%4
% z =sum(schwefel222fcn(x)) ;%5
%  z =sum(sumsquaresfcn(x)) ;%6
%  z =(brownfcn(x)) ;%7
%  z =sum(rastriginfcn(x)) ;%8
 % z =sum(griewankfcn(x)) ;%9
% z = sum(ackleyfcn(x));%10
%z = sum(exponentialfcn(x));%11
 %z =sum(periodicfcn(x)) ;%12
 %z =sum(schwefelfcn(x)) ;%13
%  z =sum(powellsumfcn(x)) ;%14
 %z = sum(salomonfcn(x));%15
% z =sum( shubert3fcn(x)) ;%16
%   z =sum(-x.*sin(sqrt(abs(x)))) ;%17
% z =sum(styblinskitankfcn(x)) ;%18
 %z =sum(alpinen1fcn(x)) ;%19
% z = sum(happycatfcn(x, 0.5));%20
%z = sum(alpinen2fcn(x));%214444
% z = sum(xinsheyangn1fcn(x));%22
% z = sum(xinsheyangn2fcn(x));%23
%z = sum(xinsheyangn3fcn(x,15,5));%24
%z = sum(xinsheyangn4fcn(x));%25
 %z= sum(zakharovfcn(x));%26
%   z = sum(ackleyfcn4(x));%27 
  %z = sum(qing(x));%28
 % z = sum(Ridgefnc(x,0.2,0.5));%29
 % z =sum(schwefel221fcn(x)) ;%30
% z =sum(schwefel220fcn(x)) ;%31
 %z =sum(schwefel223fcn(x)) ;%32


% F1


%  z=sum(x.^2);

%
% % F2
%
% z=sum(abs(x))+prod(abs(x));

%
% % F3
%

% dimension=size(x,2);
% z=0;
% for i=1:dimension
%     z=sol.fit+sum(x(1:i))^2;
% end

%
% % F4
%

% z=max(abs(x));

%
% % F5

% dimension=size(x,2);
% z=sum(100*(x(2:dimension)-(x(1:dimension-1).^2)).^2+(x(1:dimension-1)-1).^2);

%
% % F6
%

% z=sum(abs((x+.5)).^2);

%
% % F7

% dimension=size(x,2);
% z=sum([1:dimension].*(x.^4))+rand;

% % F8
%

%  z=sum(-x.*sin(sqrt(abs(x))));

%
% % F9
%

%  dimension=size(x,2);
%  z=sum(x.^2-10*cos(2*pi.*x))+10*dimension;

%
% % F10
%

% dimension=size(x,2);
% z=-20*exp(-.2*sqrt(sum(x.^2)/dimension))-exp(sum(cos(2*pi.*x))/dimension)+20+exp(1);
%
% %
% % F11
%

%  dimension=size(x,2);
% z=sum(x.^2)/4000-prod(cos(x./sqrt([1:dimension])))+1;
%
% %
% % F12
%
%
% dimension=size(x,2);
% z=(pi/dimension)*(10*((sin(pi*(1+(x(1)+1)/4)))^2)+sum((((x(1:dimension-1)+1)./4).^2).*...
% (1+10.*((sin(pi.*(1+(x(2:dimension)+1)./4)))).^2))+((x(dimension)+1)/4)^2)+sum(Ufun(x,10,100,4));

%
% % F13
%

% dimension=size(x,2);
% z=.1*((sin(3*pi*x(1)))^2+sum((x(1:dimension-1)-1).^2.*(1+(sin(3.*pi.*x(2:dimension))).^2))+...
% ((x(dimension)-1)^2)*(1+(sin(2*pi*x(dimension)))^2))+sum(Ufun(x,5,100,4));
%
%

%
% % %
% % F15
%

% aK=[.1957 .1947 .1735 .16 .0844 .0627 .0456 .0342 .0323 .0235 .0246];
% bK=[.25 .5 1 2 4 6 8 10 12 14 16];bK=1./bK;
% z=sum((aK-((x(1).*(bK.^2+x(2).*bK))./(bK.^2+x(3).*bK+x(4)))).^2);
%
% %
% % F16
%

%  z=4*(x(1)^2)-2.1*(x(1)^4)+(x(1)^6)/3+x(1)*x(2)-4*(x(2)^2)+4*(x(2)^4);

%
% % F17

% z=(x(2)-(x(1)^2)*5.1/(4*(pi^2))+5/pi*x(1)-6)^2+10*(1-1/(8*pi))*cos(x(1))+10;

%
% % F18
%

% z=(1+(x(1)+x(2)+1)^2*(19-14*x(1)+3*(x(1)^2)-14*x(2)+6*x(1)*x(2)+3*x(2)^2))*...
%      (30+(2*x(1)-3*x(2))^2*(18-32*x(1)+12*(x(1)^2)+48*x(2)-36*x(1)*x(2)+27*(x(2)^2)));
%
%

%
%

end


function R=Ufun(x,a,k,m)
R=k.*((x-a).^m).*(x>a)+k.*((-x-a).^m).*(x<(-a));
end


function x=CB(x,VarMin,VarMax)

x=max(x,VarMin);
x=min(x,VarMax);

end
