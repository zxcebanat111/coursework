function sol=fitness(sol,model)

x=sol.Position;
Dis=model.d;
N=model.n;

[~,x]=sort(x);% Random Key
x=[x x(1)];
Z=0;
for k=1:N
    i=x(k);
    j=x(k+1);
    Z=Z+Dis(i,j);
end


sol.Cost=Z;



end
