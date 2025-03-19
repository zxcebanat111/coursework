clc
clear
close all
format shortG


%% Insert Data
model=CreateModelRealData();    % Create Problem Model
CostFunction=@(tour) TourLength(tour,model);    % Cost Function

%% parametres setting
nPop=50;     % number of population
MaxIt=50;
nAlg = 3;
RLAlpha = 0;
nVar=model.n;


% GA parameters
pc=0.8;       % percent of crossover
nCross=2*round(nPop*pc/2);  % number of crossover offspring
pm=0.2;        %  percent of mutation
nMut=round(nPop*pm);  % number of mutation offspring
model.nCross = nCross;
model.nMut = nMut;



% SA Parameters
MaxSubIt=10;    % Maximum Number of Sub-iterations
T0=0.025;       % Initial Temp.
alpha=0.99;     % Temp. Reduction Rate
nMove=5;        % Number of Neighbors per Individual

% ACO Parameters
npop=50;        % Number of Ants (Population Size)
Q=1;
tau0=10*Q/(nVar*mean(model.d(:)));	% Initial Phromone
alpha=1;        % Phromone Exponential Weight
beta=1;         % Heuristic Exponential Weight
rho=0.05;       % Evaporation Rate
eta=1./model.d;             % Heuristic Information Matrix
tau=tau0*ones(nVar,nVar);   % Phromone Matrix



% ICA Parameters
nVar=model.n;              % Number of Decision Variables
lb=0*ones(1,nVar); % Lower Bound of Variables
ub=1*ones(1,nVar); % Upper Bound of Variables



ncountries=50;     % Population Size
nimp=10;            % Number of Empires/Imperialists
MaxIt=50;        % Maximum Number of Iterations


beta=2;             % Assimilation Coefficient
P_revolve=0.1;      % Revolution Probability
zeta=0.1;           % Colonies Mean Cost Coefficient

model.Size.Position=[1 nVar];   % Decision Variables Matrix Size
model.nvar=nVar;
model.lb=lb;
model.ub=ub;
model.ncountries=ncountries;
model.nimp=nimp;
model.beta=beta;
model.P_revolve=P_revolve;
model.zeta=zeta;

% Tabo parameters
TL=round(0.5*nPop);      % Tabu Length
TC=zeros(nPop,1);




%% initialization



% Create Empty Structure for Individuals
empty_individual.Position=[];
empty_individual.Cost=[];

% Create Population Array
pop=repmat(empty_individual,nPop,1);

% Initialize Best Solution
BestSol.Cost=inf;

% Initialize Population
for i=1:nPop
    
    % Initialize Position
    pop(i).Position=CreateRandomSolution(model);
    
    % Evaluation
    pop(i).Cost=CostFunction(pop(i).Position);
    
    % Update Best Solution
    if pop(i).Cost<=BestSol.Cost
        BestSol=pop(i);
    end
    
end

% Array to Hold Best Cost Values
BestCostMaxiter = zeros(MaxIt,1);
BestCost=zeros(nAlg,1);
names = {'sa','ga','aco','ts','ica'};
Alg.name = [];
Alg.rewards = 0;
Alg.value = [];
algRewards = repmat(Alg,nAlg,1);
for i=1:nAlg
    algRewards(i).name = names(i);
    algRewards(i).value = i;
end
data.pop=pop;
model.pop = pop;
lotchList = [];
T=T0;

for it=1:nAlg
   pop = data.pop;
   
   
  switch it
      case 1
             
    for subit=1:MaxSubIt
        
        % Create and Evaluate New Solutions
        newpop=repmat(empty_individual,nPop,nMove);
        for i=1:nPop
            for j=1:nMove
                
                % Create Neighbor
                newpop(i,j).Position=CreateNeighbor(pop(i).Position);
                
                % Evaluation
                newpop(i,j).Cost=CostFunction(newpop(i,j).Position);
                
            end
        end
        newpop=newpop(:);
        
        % Sort Neighbors
        [~, SortOrder]=sort([newpop.Cost]);
        newpop=newpop(SortOrder);
        
        for i=1:nPop
            
            if newpop(i).Cost<=pop(i).Cost
                pop(i)=newpop(i);
                
            else
                DELTA=(newpop(i).Cost-pop(i).Cost)/pop(i).Cost;
                P=exp(-DELTA/T);
                if rand<=P
                    pop(i)=newpop(i);
                end
            end
            
            % Update Best Solution Ever Found
            if pop(i).Cost<=BestSol.Cost
                BestSol=pop(i);
            end
        
        end

    end
    
    % Store Best Cost Ever Found
    algRewards(1).rewards =1/BestSol.Cost;
    BestCost(it)=BestSol.Cost;
    
          
          
      case 2
          % crossover
    crosspop=repmat(empty_individual,nCross,1);
    crosspop=crossover(crosspop,pop,model);
    
   for i=1:nCross
    
    crosspop(i).Cost = TourLength(crosspop(i).Position,model);
   end
    
    
    % mutation
    mutpop=repmat(empty_individual,nMut,1);
    mutpop=mutation(mutpop,pop,model);
   for i=1:nMut
    
      mutpop(i).Cost = CostFunction(mutpop(i).Position);
   end
    
    
    % Merged
    [pop]=[pop;crosspop;mutpop];
    
    
   

    % Sorting
    [value,index]=sort([pop.Cost]);
    pop=pop(index);
    gpop=pop(1);
    
    % Select
    pop=pop(1:nPop);
    

    BestCost(it)=gpop.Cost;
    algRewards(2).rewards =1/gpop.Cost;
          
        
    
          
  
          
      case 3
          
        % Move Ants
    for k=1:npop
        
        pop(k).Position=randi([1 nVar]);
        
        for l=2:nVar
            
            i=pop(k).Position(end);
            
            P=tau(i,:).^alpha.*eta(i,:).^beta;
            
            P(pop(k).Position)=0;
            
            P=P/sum(P);
            
            j=RouletteWheelSelection(P);
            
            pop(k).Position=[pop(k).Position j];
            
        end
        
        pop(k).Cost=CostFunction(pop(k).Position);
        
        if pop(k).Cost<BestSol.Cost
            BestSol=pop(k);
        end
        
    end
    
    % Update Phromones
    for k=1:npop
        
        tour=pop(k).Position;
        
        tour=[tour tour(1)]; %#ok
        
        for l=1:nVar
            
            i=tour(l);
            j=tour(l+1);
            
            tau(i,j)=tau(i,j)+Q/pop(k).Cost;
            
        end
        
    end
    
    % Evaporation
    tau=(1-rho)*tau;
    
    % Store Best Cost
    algRewards(1).rewards =1/BestSol.Cost;
    BestCost(it)=BestSol.Cost;
    
     
   
  end
  disp(['not Iter ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
  T=alpha*T; 
end
[~,index] = sortrows([algRewards.rewards].');
algRewards = algRewards(index(end:-1:1));
clear index
Algs = (1:nAlg);
tic
BestSolPerIter = pop(50);

for iter=1:MaxIt
   chooseRandomic = rand(1);

if iter<3
    lotCH=iter;
elseif chooseRandomic > RLAlpha 
    numi = randi([1,3]);
    if numi==1
        lotCH = algRewards(1).value;
    elseif numi==2
        lotCH = algRewards(2).value;
    elseif numi==3
        lotCH = algRewards(3).value;
    end
elseif chooseRandomic < RLAlpha 
    lotCH = algRewards(1).value;
 else
        numi = randi([2,3]);
   if numi==2
        lotCH = algRewards(2).value;
        else
        lotCH = algRewards(3).value;
   end  
end
lotchList = [lotchList,lotCH];
    
    %% -------------------------------------------------------------
  
 switch lotCH
  
      case 1
             
    for subit=1:MaxSubIt
        
        %Create and Evaluate New Solutions
        newpop=repmat(empty_individual,nPop,nMove);
        for i=1:nPop
            for j=1:nMove
                
                % Create Neighbor
                newpop(i,j).Position=CreateNeighbor(pop(i).Position);
                
                % Evaluation
                newpop(i,j).Cost=CostFunction(newpop(i,j).Position);
                
            end
        end
        newpop=newpop(:);
        [newpop] = [newpop;BestSolPerIter];
        
        % Sort Neighbors
        [~, SortOrder]=sort([newpop.Cost]);
        newpop=newpop(SortOrder);
        
        for i=1:nPop
            
            if newpop(i).Cost<=pop(i).Cost
                pop(i)=newpop(i);
                
            else
                DELTA=(newpop(i).Cost-pop(i).Cost)/pop(i).Cost;
                P=exp(-DELTA/T);
                if rand<=P
                    pop(i)=newpop(i);
                end
            end
            
            % Update Best Solution Ever Found
            if pop(i).Cost<=BestSol.Cost
                BestSol=pop(i);
            end
        
        end

    end
    
    % Store Best Cost Ever Found
    BestCostMaxiter(iter)=BestSol.Cost;
    
     if iter > 1
    algRewards(1).rewards =algRewards(1).rewards+ (BestCostMaxiter(iter-1)-BestCostMaxiter(iter));
    else
            algRewards(1).rewards =algRewards(1).rewards+ (BestCostMaxiter(it));
    end
    
          
          
      case 2
          % crossover
    crosspop=repmat(empty_individual,nCross,1);
    crosspop=crossover(crosspop,pop,model);
    
   for i=1:nCross
    
    crosspop(i).Cost = TourLength(crosspop(i).Position,model);
   end
    
    
    % mutation
    mutpop=repmat(empty_individual,nMut,1);
    mutpop=mutation(mutpop,pop,model);
   for i=1:nMut
    
      mutpop(i).Cost = CostFunction(mutpop(i).Position);
   end
    
    
    % Merged
    [pop]=[pop;crosspop;mutpop;BestSolPerIter];
    
    
   

    % Sorting
    [value,index]=sort([pop.Cost]);
    pop=pop(index);
    gpop=pop(1);
    
    % Select
    pop=pop(1:nPop);
    

    BestCostMaxiter(iter)=gpop.Cost;
    
     if iter > 1
    algRewards(2).rewards =algRewards(2).rewards+ (BestCostMaxiter(iter-1)-BestCostMaxiter(iter));
    else
            algRewards(2).rewards =algRewards(2).rewards+ (BestCostMaxiter(it));
    end
          
        
    
          
  
          
      case 3
          
        % Move Ants
    for k=1:npop
        
        pop(k).Position=randi([1 nVar]);
        
        for l=2:nVar
            
            i=pop(k).Position(end);
            
            P=tau(i,:).^alpha.*eta(i,:).^beta;
            
            P(pop(k).Position)=0;
            
            P=P/sum(P);
            
            j=RouletteWheelSelection(P);
            
            pop(k).Position=[pop(k).Position j];
            
        end
        
        pop(k).Cost=CostFunction(pop(k).Position);
        
        if pop(k).Cost<BestSol.Cost
            BestSol=pop(k);
        end
        
    end
    
    % Update Phromones
    for k=1:npop
        
        tour=pop(k).Position;
        
        tour=[tour tour(1)]; %#ok
        
        for l=1:nVar
            
            i=tour(l);
            j=tour(l+1);
            
            tau(i,j)=tau(i,j)+Q/pop(k).Cost;
            
        end
        
    end
    
    % Evaporation
    tau=(1-rho)*tau;
    
    % Store Best Cost
    
    BestCostMaxiter(iter)=BestSol.Cost;
     if iter > 1
    algRewards(3).rewards =algRewards(3).rewards+ (BestCostMaxiter(iter-1)-BestCostMaxiter(iter));
    else
            algRewards(3).rewards =algRewards(3).rewards+ (BestCostMaxiter(it));
    end
    
     
   
 end
 
 randomi = randi([1,2]);
 switch randomi
     
     case 1
 
 
 imp=CreateInitialPopulation(model);

[value,index]=min([imp.Cost]);
gimp=imp(index); % gimp = Best Of Solution
% Assimilation
    imp=Assimilation(imp,model);
    
    % Revolution
    imp=Revolution(imp,model);
    
    % Exchange
    imp=Exchange(imp);
    
    % Totla Fitness
    imp=CalTotalFitness(imp,model);
    
    % Imperialistic Competition
    imp=ImperialisticCompetition(imp);
    
    
    [value,index]=min([imp.Cost]);
    
    if value<gimp.Cost
        gimp=imp(index);
    end
    if gimp.Cost>BestSolPerIter.Cost
        gimp = BestSolPerIter;
    end

    
    BestCostMaxiter(iter)=gimp.Cost;
    
    
    nimp=length(imp);




     case 2

 bestnewsol.Cost=inf;
    
    % Apply Actions
    for i=1:nPop
        if TC(i)==0
            newsol.Position=pop(i).Position;
            newsol.Cost=pop(i).Cost;
            newsol.ActionIndex=i;

            if newsol.Cost<=bestnewsol.Cost
                bestnewsol=newsol;
            end
        end
    end
    
    % Update Current Solution
    sol=bestnewsol;
    
    % Update Tabu List
    for i=1:nPop
        if i==bestnewsol.ActionIndex
            TC(i)=TL;               % Add To Tabu List
        else
            TC(i)=max(TC(i)-1,0);   % Reduce Tabu Counter
        end
    end
    
    % Update Best Solution Ever Found
    if sol.Cost<=BestSol.Cost
        BestSol=sol;
    end
    if BestSol.Cost>BestSolPerIter.Cost
        BestSol = BestSolPerIter;
    end

    
    % Save Best Cost Ever Found
    BestCostMaxiter(iter)=BestSol.Cost;

 end

  
 
 
 
  RLAlpha = RLAlpha+nAlg/MaxIt; 
[~,index] = sortrows([algRewards.rewards].');
algRewards = algRewards(index(end:-1:1));
clear index
 BestSolPerIterT = BestSol;
 BestSolPerIter.Position=BestSolPerIterT.Position;
 BestSolPerIter.Cost=BestSolPerIterT.Cost;
 disp(['Iteration ' num2str(iter) ': Best Cost = ' num2str(BestCostMaxiter(iter) )]);
  T=alpha*T; 
end
toc
figure;
%plot(BestCost,'LineWidth',2);
semilogy(BestCostMaxiter,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;


