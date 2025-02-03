function  imp=CalTotalFitness(imp,model)

zeta=model.zeta;
nimp=numel(imp);

for i=1:nimp
   
    imp(i).totalfit=imp(i).Cost+zeta*mean([imp(i).colony.Cost]);
    
end






end







