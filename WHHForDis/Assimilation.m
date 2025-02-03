function  imp=Assimilation(imp,model)

beta=model.beta;
nimp=numel(imp);
Size=model.Size;

for i=1:nimp
    
    ncolony=length(imp(i).colony);
    
    for j=1:ncolony
    
        d=imp(i).Position-imp(i).colony(j).Position;
        d=d.*rand(Size.Position)*beta;
        imp(i).colony(j).Position=imp(i).colony(j).Position+d;
        imp(i).colony(j)=fitness(imp(i).colony(j),model);
        
        
    end
end






end







