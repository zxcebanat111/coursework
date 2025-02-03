function  imp=Revolution(imp,model)


P_revolve=model.P_revolve;
nimp=numel(imp);
Size=model.Size;
nvar=model.nvar;

for i=1:nimp

    ncolony=length(imp(i).colony);
    
    for j=1:ncolony
         
        
        if rand<P_revolve
        k=randsample(nvar,ceil(0.05*nvar));   
        d=model.ub-model.lb;
        d=0.1*randn(Size.Position).*d;
        imp(i).colony(j).Position(k)=imp(i).colony(j).Position(k)+d(k);
%         dd=unifrnd(data.lb,data.ub);imp(i).colony(j).x(k)=dd(k);
        imp(i).colony(j)=fitness(imp(i).colony(j),model);
        end
        
        
    end
end






end







