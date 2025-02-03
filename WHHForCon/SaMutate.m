
function y=Mutate(x,mu,sigma,VarMin,VarMax)

    A=(rand(size(x))<=mu);
    J=find(A==1);

    y=x;
    y(J)=x(J)+sigma*randn(size(J));

    % Clipping
    y=max(y,VarMin);
    y=min(y,VarMax);
    
end