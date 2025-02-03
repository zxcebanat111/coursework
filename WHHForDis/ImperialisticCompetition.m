function  imp=ImperialisticCompetition(imp)


if numel(imp)==1
    return;
end
    
    
[~,index1]=max([imp.totalfit]);
wimp=imp(index1);

[~,index2]=max([wimp.colony.Cost]);
wcolony=wimp.colony(index2);


imp(index1).colony=imp(index1).colony([1:index2-1 index2+1:end]);


P=[imp.totalfit];
P=1./P;
P(index1)=0;
k=RouletteWheel(P);

n=length(imp(k).colony);
imp(k).colony(n+1).Position=wcolony.Position;
imp(k).colony(n+1).Cost=wcolony.Cost;
%imp(k).colony(n+1).info=wcolony.info;



n=length(imp(index1).colony);

if n==0
    
imp=imp([1:index1-1 index1+1:end]);
P=[imp.totalfit];
P=1./P;
k=RouletteWheel(P);

n=length(imp(k).colony);
imp(k).colony(n+1).Position=wimp.Position;
imp(k).colony(n+1).Cost=wimp.Cost; 
%imp(k).colony(n+1).info=wimp.info; 
  
end












end







