function  crosspop=crossover(crosspop,pop,model)

ncross=model.nCross;

f=[pop.Cost];
f=1./f;
f=f./sum(f);
f=cumsum(f);

for n=1:2:ncross
    
    i1=find(rand<=f,1,'first');
    i2=find(rand<=f,1,'first');
    
    [crosspop(n).Position,crosspop(n+1).Position]=SinglePointCrossover(pop(i1).Position,pop(i2).Position);
    
    
%     crosspop(n).Cost = TourLength(crosspop(n).Position,model);
%     crosspop(n+1).Cost = TourLength(crosspop(n+1).Position,model);

    
    
end

end


function [y1,y2]=SinglePointCrossover(x1,x2)

nvar=numel(x1);

j=randi([1 nvar-1]);
y1=x1;
y2=x2;

y1(1:j)=x2(1:j);
y2(1:j)=x1(1:j);

y1=Unique(y1,1:j);
y2=Unique(y2,1:j);

end


















