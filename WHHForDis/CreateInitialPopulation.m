function imp=CreateInitialPopulation(model)

nimp=model.nimp;
ncountries=model.ncountries;

emp.Position=[];
emp.Cost=[];

colony=repmat(emp,ncountries,1);

colony=model.pop;
for i=1:ncountries
    
    colony(i)=fitness(colony(i),model);
end



[~,ind]=sort([colony.Cost]);

colony=colony(ind);

imp=colony(1:nimp);

colony=colony(nimp+1:end);


ncolony=length(colony);
colony=colony(randperm(ncolony));




k=0;
j=1;
for i=1:ncolony
    k=k+1;
    imp(k).colony(j)=colony(i);

    if k==nimp
        k=0;
        j=j+1;
    end  
end


end