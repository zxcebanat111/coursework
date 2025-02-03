function  imp=Exchange(imp)

nimp=numel(imp);

for i=1:nimp
    
    [value,index]=min([imp(i).colony.Cost]);
    
    if value<imp(i).Cost
        
        bestcolony=imp(i).colony(index);
        
        imp(i).colony(index).Position=imp(i).Position;
        imp(i).colony(index).Cost=imp(i).Cost;
        %imp(i).colony(index).info=imp(i).info;
        
        imp(i).Position=bestcolony.Position;
        imp(i).Cost=bestcolony.Cost;
        %imp(i).info=bestcolony.info;
        
        
        
    end
    
end






end







