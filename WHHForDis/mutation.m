function  mutpop=mutation(mutpop,pop,model)

nmut=model.nMut;
npop=numel(pop);


for n=1:nmut
    
    i1=randi([1 npop]);
    mutpop(n).Position=Swap(pop(i1).Position);
%     mutpop(n).Cost = CostFunction(mutpop(n).Position);
    
    
end


end





function y=Swap(x)

n=numel(x);

i=randsample(n,2);
i1=i(1);
i2=i(2);

y=x;
y([i1 i2])=x([i2 i1]);

end


