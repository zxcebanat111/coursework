clc; 
clear;
close all;

%% Problem Definition

CostFunction=@(x) Sphere(x);        % Cost Function

nVar=5;             % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size

VarMin=-10;         % Decision Variables Lower Bound
VarMax= 10;         % Decision Variables Upper Bound

%% ALL Parameters
% origin parameters
MaxIt=1000;        
nPop=50;
nAlg = 10;
RLAlpha = 10;

% BBO parameters
KeepRate=0.4;                   % Keep Rate
nKeep=round(KeepRate*nPop);     % Number of Kept Habitats
bbonNew=nPop-nKeep;                % Number of New Habitats
bboMu=linspace(1,0,nPop);          % Emmigration Rates
lambda=1-bboMu;                    % Immigration Rates
bboAlpha=0.9;                      % alpha value
pMutation=0.1;                  % PMutation value
bboSigma=0.02*(VarMax-VarMin);     %sigma value


% DE parameters
beta_min=0.2;   % Lower Bound of Scaling Factor
beta_max=0.8;   % Upper Bound of Scaling Factor
pCR=0.2;        % Crossover Probability

% FireFly parameters
ffGamma=1;            % Light Absorption Coefficient
beta0=2;            % Attraction Coefficient Base Value
ffAlpha=0.2;          % Mutation Coefficient
alpha_damp=0.98;    % Mutation Coefficient Damping Ratio
delta=0.05*(VarMax-VarMin);     % Uniform Mutation Range
m=2;
if isscalar(VarMin) && isscalar(VarMax)
    dmax = (VarMax-VarMin)*sqrt(nVar);
else
    dmax = norm(VarMax-VarMin);
end

% GA parameters
pc=0.8;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (also Parnets)
gaGamma=0.4;              % Extra Range Factor for Crossover
pm=0.2;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants
gaBeta=8; % Selection Pressure
gaMu = 0.1;

% HS parameters
HMS=50;         % Harmony Memory Size
nNew=20;        % Number of New Harmonies
HMCR=0.9;       % Harmony Memory Consideration Rate
PAR=0.1;        % Pitch Adjustment Rate
FW=0.02*(VarMax-VarMin);    % Fret Width (Bandwidth)
FW_damp=0.995;              % Fret Width Damp Ratio


% PSO parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

% SA parameters
MaxSubIt=20;    % Maximum Number of Sub-iterations
T0=0.1;       % Initial Temp.
saAlpha=0.99;     % Temp. Reduction Rate
nMove=5;        % Number of Neighbors per Individual
saMu = 0.5;       % Mutation Rate
saSigma = 0.1*(VarMax-VarMin);    % Mutation Range (Standard Deviation)

% SFLA parameters
nPopMemeplex = 10;                          % Memeplex Size
nPopMemeplex = max(nPopMemeplex, nVar+1);   % Nelder-Mead Standard
nMemeplex = 5;                  % Number of Memeplexes
I = reshape(1:nPop, nMemeplex, []);
fla_params.q = max(round(0.3*nPopMemeplex),2);   % Number of Parents
fla_params.alpha = 3;   % Number of Offsprings
fla_params.beta = 5;    % Maximum Number of Iterations
fla_params.sigma = 2;   % Step Size
fla_params.CostFunction = CostFunction;
fla_params.VarMin = VarMin;
fla_params.VarMax = VarMax;

% IWO paramaters
Smin = 0;       % Minimum Number of Seeds
Smax = 5;       % Maximum Number of Seeds
Exponent = 2;           % Variance Reduction Exponent
sigma_initial = 0.5;    % Initial Value of Standard Deviation
sigma_final = 0.001;	% Final Value of Standard Deviation





%% Initialization
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Velocity=[];
empty_individual.Best.Position=[];
empty_individual.Best.Cost=[];

BestSol.Cost=inf;
pop=repmat(empty_individual,nPop,1);

% Initialize Habitats
for i=1:nPop
    
    % Initialize Position
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    pop(i).Velocity=zeros(VarSize);
    
    % Evaluation
    pop(i).Cost=CostFunction(pop(i).Position);
    
    % Update Personal Best
    pop(i).Best.Position=pop(i).Position;
    pop(i).Best.Cost=pop(i).Cost;
    
    % Update Global Best
    if pop(i).Best.Cost<BestSol.Cost
        
        GlobalBest=pop(i).Best;
        
    end
    
end

% Sort Population
[~, SortOrder]=sort([pop.Cost]);
pop=pop(SortOrder);
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);

% Best Solution Ever Found
BestSol=pop(1);

% Array to Hold Best Costs
BestCostMaxiter = zeros(MaxIt,1);
BestCost=zeros(nAlg,1);
WorstCost=pop(end).Cost;
T=T0;

%%----------------------------
names = {'bbo','de','fa','hs','pso','sa','sfla','tlbo','iwo','ga'};
Alg.name = [];
Alg.rewards = 0;
Alg.value = [];
algRewards = repmat(Alg,nAlg,1);
for i=1:nAlg
    algRewards(i).name = names(i);
    algRewards(i).value = i;
end

data.pop=pop;
lotchList = [];

%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% =============================== main part of Algorithm ===========================
for it=1:nAlg
    pop = data.pop;
    
    
    switch it
        case 1
            
            newpop=pop;
            for i=1:nPop
                for k=1:nVar
                    % Migration
                    if rand<=lambda(i)
                        % Emmigration Probabilities
                        EP=bboMu;
                        EP(i)=0;
                        EP=EP/sum(EP);
                        
                        % Select Source Habitat
                        j=RouletteWheelSelection(EP);
                        
                        % Migration
                        newpop(i).Position(k)=pop(i).Position(k) ...
                            +bboAlpha*(pop(j).Position(k)-pop(i).Position(k));
                        
                    end
                    
                    % Mutation
                    if rand<=pMutation
                        newpop(i).Position(k)=newpop(i).Position(k)+bboSigma*randn;
                    end
                end
                
                % Apply Lower and Upper Bound Limits
                newpop(i).Position = max(newpop(i).Position, VarMin);
                newpop(i).Position = min(newpop(i).Position, VarMax);
                
                % Evaluation
                newpop(i).Cost=CostFunction(newpop(i).Position);
            end
            
            % Sort New Population
            [~, SortOrder]=sort([newpop.Cost]);
            newpop=newpop(SortOrder);
            
            % Select Next Iteration Population
            pop=[pop(1:nKeep)
                newpop(1:bbonNew)];
            
            % Sort Population
            [~, SortOrder]=sort([pop.Cost]);
            pop=pop(SortOrder);
            
            % Update Best Solution Ever Found
            BestSol=pop(1);
            
            
          rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(1).rewards =1/rewardTemp;
            else
                algRewards(1).rewards =abs(rewardTemp);
          end
            
            % Store Best Cost Ever Found
            BestCost(1)=BestSol.Cost;
          
            
            
        case 2
            
            
            for ide=1:nPop
                
                x=pop(ide).Position;
                
                A=randperm(nPop);
                
                A(A==ide)=[];
                
                a=A(1);
                b=A(2);
                c=A(3);
                
                % Mutation
                %beta=unifrnd(beta_min,beta_max);
                beta=unifrnd(beta_min,beta_max,VarSize);
                y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);
                y = max(y, VarMin);
                y = min(y, VarMax);
                
                % Crossover
                z=zeros(size(x));
                j0=randi([1 numel(x)]);
                for j=1:numel(x)
                    if j==j0 || rand<=pCR
                        z(j)=y(j);
                    else
                        z(j)=x(j);
                    end
                end
                
                newpop(it).Position=z;
                newpop(it).Cost=CostFunction(newpop(it).Position);
                
                if newpop(it).Cost<pop(ide).Cost
                    pop(ide)=newpop(it);
                    
                    if pop(ide).Cost<BestSol.Cost
                        BestSol=pop(ide);
                    end
                end
                
            end
            
            % Update Best Cost
            BestCost(2)=BestSol.Cost;
           rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(2).rewards =1/rewardTemp;
            else
                algRewards(2).rewards =abs(rewardTemp);
          end
            
            
            
            
            
        case 3
            
            temppop=repmat(empty_individual,nPop,1);
            for i=1:nPop
                temppop(i).Cost = inf;
                for j=1:nPop
                    if pop(j).Cost < pop(i).Cost
                        rij=norm(pop(i).Position-pop(j).Position)/dmax;
                        beta=beta0*exp(-ffGamma*rij^m);
                        e=delta*unifrnd(-1,+1,VarSize);
                        %e=delta*randn(VarSize);
                        
                        newpop(i).Position = pop(i).Position ...
                            + beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
                            + ffAlpha*e;
                        
                        newpop(i).Position=max(newpop(i).Position,VarMin);
                        newpop(i).Position=min(newpop(i).Position,VarMax);
                        
                        newpop(i).Cost=CostFunction(newpop(i).Position);
                        
                        if newpop(i).Cost <= temppop(i).Cost
                            temppop(i) = newpop(i);
                            if temppop(i).Cost<=BestSol.Cost
                                BestSol=temppop(i);
                            end
                        end
                        
                    end
                end
            end
            
            % Merge
            pop=[pop
                temppop];  %#ok
            
            % Sort
            [~, SortOrder]=sort([pop.Cost]);
            pop=pop(SortOrder);
            
            % Truncate
            pop=pop(1:nPop);
            
            % Store Best Cost Ever Found
            BestCost(3)=BestSol.Cost;
            
            ffAlpha = ffAlpha*alpha_damp;
            
            % Store Best Cost Ever Found
           rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(3).rewards =1/rewardTemp;
            else
                algRewards(3).rewards =abs(rewardTemp);
          end
            
            
            
        case 4
            
            
            % Initialize Array for New Harmonies
            NEW=repmat(empty_individual,nNew,1);
            
            % Create New Harmonies
            for k=1:nNew
                
                % Create New Harmony Position
                NEW(k).Position=unifrnd(VarMin,VarMax,VarSize);
                for j=1:nVar
                    if rand<=HMCR
                        % Use Harmony Memory
                        i=randi([1 HMS]);
                        NEW(k).Position(j)=pop(i).Position(j);
                    end
                    
                    % Pitch Adjustment
                    if rand<=PAR
                        %DELTA=FW*unifrnd(-1,+1);    % Uniform
                        DELTA=FW*randn();            % Gaussian (Normal)
                        NEW(k).Position(j)=NEW(k).Position(j)+DELTA;
                    end
                    
                end
                
                % Apply Variable Limits
                NEW(k).Position=max(NEW(k).Position,VarMin);
                NEW(k).Position=min(NEW(k).Position,VarMax);
                
                % Evaluation
                NEW(k).Cost=CostFunction(NEW(k).Position);
                
            end
            
            % Merge Harmony Memory and New Harmonies
            pop=[pop
                NEW]; %#ok
            
            % Sort Harmony Memory
            [~, SortOrder]=sort([pop.Cost]);
            pop=pop(SortOrder);
            
            % Truncate Extra Harmonies
            pop=pop(1:HMS);
            
            % Update Best Solution Ever Found
            BestSol=pop(1);
            FW=FW*FW_damp;
            % Store Best Cost Ever Found
          rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(4).rewards =1/rewardTemp;
            else
                algRewards(4).rewards =abs(rewardTemp);
          end
            
            
            
            
        case 5
            
            for i=1:nPop
                
                % Update Velocity
                pop(i).Velocity = w*pop(i).Velocity ...
                    +c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
                    +c2*rand(VarSize).*(GlobalBest.Position-pop(i).Position);
                
                % Apply Velocity Limits
                pop(i).Velocity = max(pop(i).Velocity,VelMin);
                pop(i).Velocity = min(pop(i).Velocity,VelMax);
                
                % Update Position
                pop(i).Position = pop(i).Position + pop(i).Velocity;
                
                % Velocity Mirror Effect
                IsOutside=(pop(i).Position<VarMin | pop(i).Position>VarMax);
                pop(i).Velocity(IsOutside)=-pop(i).Velocity(IsOutside);
                
                % Apply Position Limits
                pop(i).Position = max(pop(i).Position,VarMin);
                pop(i).Position = min(pop(i).Position,VarMax);
                
                % Evaluation
                pop(i).Cost = CostFunction(pop(i).Position);
                
                % Update Personal Best
                if pop(i).Cost<pop(i).Best.Cost
                    
                    pop(i).Best.Position=pop(i).Position;
                    pop(i).Best.Cost=pop(i).Cost;
                    
                    % Update Global Best
                    if pop(i).Best.Cost<GlobalBest.Cost
                        
                        GlobalBest=pop(i).Best;
                        
                    end
                    
                end
                
            end
            w=w*wdamp;
            BestCost(5)=GlobalBest.Cost;
            rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(5).rewards =1/rewardTemp;
            else
                algRewards(5).rewards =abs(rewardTemp);
          end
            
            
        case 6
            
            
            for subit=1:MaxSubIt
                
                % Create and Evaluate New Solutions
                newpop=repmat(empty_individual,nPop,nMove);
                for i=1:nPop
                    for j=1:nMove
                        
                        % Create Neighbor
                        newpop(i,j).Position=SaMutate(pop(i).Position,saMu,saSigma,VarMin,VarMax);
                        
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
                    if pop(i).Cost<=1/BestSol.Cost
                        BestSol=pop(i);
                    end
                    
                end
                
            end
            
            % Store Best Cost Ever Found
            BestCost(6)=BestSol.Cost;
          rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(6).rewards =1/rewardTemp;
            else
                algRewards(6).rewards =abs(rewardTemp);
          end
            T=saAlpha*T;
            saSigma = 0.98*saSigma;
            
            
            
        case 7
            
            
            
            fla_params.BestSol = BestSol;
            
            % Initialize Memeplexes Array
            Memeplex = cell(nMemeplex, 1);
            
            % Form Memeplexes and Run FLA
            for j = 1:nMemeplex
                % Memeplex Formation
                Memeplex{j} = pop(I(j,:));
                
                % Run FLA
                Memeplex{j} = RunFLA(Memeplex{j}, fla_params);
                
                % Insert Updated Memeplex into Population
                pop(I(j,:)) = Memeplex{j};
            end
            
            % Sort Population
            pop = SortPopulation(pop);
            
            % Update Best Solution Ever Found
            BestSol = pop(1);
            
            % Store Best Cost Ever Found
            BestCost(7) = BestSol.Cost;
           rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(7).rewards =1/rewardTemp;
            else
                algRewards(7).rewards =abs(rewardTemp);
          end
            
            
            
        case 8
            
            
            % Calculate Population Mean
            Mean = 0;
            for i=1:nPop
                Mean = Mean + pop(i).Position;
            end
            Mean = Mean/nPop;
            
            % Select Teacher
            Teacher = pop(1);
            for i=2:nPop
                if pop(i).Cost < Teacher.Cost
                    Teacher = pop(i);
                end
            end
            
            % Teacher Phase
            for i=1:nPop
                % Create Empty Solution
                newsol = empty_individual;
                
                % Teaching Factor
                TF = randi([1 2]);
                
                % Teaching (moving towards teacher)
                newsol.Position = pop(i).Position ...
                    + rand(VarSize).*(Teacher.Position - TF*Mean);
                
                % Clipping
                newsol.Position = max(newsol.Position, VarMin);
                newsol.Position = min(newsol.Position, VarMax);
                
                % Evaluation
                newsol.Cost = CostFunction(newsol.Position);
                
                % Comparision
                if newsol.Cost<pop(i).Cost
                    pop(i) = newsol;
                    if pop(i).Cost < BestSol.Cost
                        BestSol = pop(i);
                    end
                end
            end
            
            % Learner Phase
            for i=1:nPop
                
                A = 1:nPop;
                A(i)=[];
                j = A(randi(nPop-1));
                
                Step = pop(i).Position - pop(j).Position;
                if pop(j).Cost < pop(i).Cost
                    Step = -Step;
                end
                
                % Create Empty Solution
                newsol = empty_individual;
                
                % Teaching (moving towards teacher)
                newsol.Position = pop(i).Position + rand(VarSize).*Step;
                
                % Clipping
                newsol.Position = max(newsol.Position, VarMin);
                newsol.Position = min(newsol.Position, VarMax);
                
                % Evaluation
                newsol.Cost = CostFunction(newsol.Position);
                
                % Comparision
                if newsol.Cost<pop(i).Cost
                    pop(i) = newsol;
                    if pop(i).Cost < BestSol.Cost
                        BestSol = pop(i);
                    end
                end
            end
            
            % Store Record for Current Iteration
            BestCost(8) = BestSol.Cost;
           rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(8).rewards =1/rewardTemp;
            else
                algRewards(8).rewards =abs(rewardTemp);
          end
            
            
            
            
            
        case 9
            
            
            
            
            % Update Standard Deviation
            sigma = ((MaxIt - it)/(MaxIt - 1))^Exponent * (sigma_initial - sigma_final) + sigma_final;
            
            % Get Best and Worst Cost Values
            Costs = [pop.Cost];
            BestCost = min(Costs);
            WorstCost = max(Costs);
            
            % Initialize Offsprings Population
            newpop = [];
            
            % Reproduction
            for i = 1:numel(pop)
                
                ratio = (pop(i).Cost - WorstCost)/(BestCost - WorstCost);
                S = floor(Smin + (Smax - Smin)*ratio);
                
                for j = 1:S
                    
                    % Initialize Offspring
                    newsol = empty_individual;
                    
                    % Generate Random Location
                    newsol.Position = pop(i).Position + sigma * randn(VarSize);
                    
                    % Apply Lower/Upper Bounds
                    newsol.Position = max(newsol.Position, VarMin);
                    newsol.Position = min(newsol.Position, VarMax);
                    
                    % Evaluate Offsring
                    newsol.Cost = CostFunction(newsol.Position);
                    
                    % Add Offpsring to the Population
                    newpop = [newpop
                        newsol];  %#ok
                    
                end
                
            end
            
            % Merge Populations
            pop = [pop
                newpop];
            
            % Sort Population
            [~, SortOrder]=sort([pop.Cost]);
            pop = pop(SortOrder);
            
            % Competitive Exclusion (Delete Extra Members)
            if numel(pop)>nPop
                pop = pop(1:nPop);
            end
            
            % Store Best Solution Ever Found
            BestSol = pop(1);
            
            
            BestCost(9) = BestSol.Cost;
            rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(9).rewards =1/rewardTemp;
            else
                algRewards(9).rewards =abs(rewardTemp);
          end
            
            
            
            
            
        case 10
            
            
            % Calculate Selection Probabilities
            %     if UseRouletteWheelSelection
            P=exp(-beta*Costs/WorstCost);
            P=P/sum(P);
            %     end
            
            % Crossover
            popc=repmat(empty_individual,nc/2,2);
            for k=1:nc/2
                
                %         % Select Parents Indices
                %         if UseRouletteWheelSelection
                i1=RouletteWheelSelection(P);
                i2=RouletteWheelSelection(P);
                %         end
                %         if UseTournamentSelection
                %             i1=TournamentSelection(pop,TournamentSize);
                %             i2=TournamentSelection(pop,TournamentSize);
                %         end
                %         if UseRandomSelection
                %             i1=randi([1 nPop]);
                %             i2=randi([1 nPop]);
                %         end
                
                % Select Parents
                p1=pop(i1);
                p2=pop(i2);
                
                % Apply Crossover
                [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position,gaGamma,VarMin,VarMax);
                
                % Evaluate Offsprings
                popc(k,1).Cost=CostFunction(popc(k,1).Position);
                popc(k,2).Cost=CostFunction(popc(k,2).Position);
                
            end
            popc=popc(:);
            
            
            % Mutation
            popm=repmat(empty_individual,nm,1);
            for k=1:nm
                
                % Select Parent
                i=randi([1 nPop]);
                p=pop(i);
                
                % Apply Mutation
                popm(k).Position=Mutate(p.Position,gaMu,VarMin,VarMax);
                
                % Evaluate Mutant
                popm(k).Cost=CostFunction(popm(k).Position);
                
            end
            
            % Create Merged Population
            pop=[pop
                popc
                popm]; %#ok
            
            % Sort Population
            Costs=[pop.Cost];
            [Costs, SortOrder]=sort(Costs);
            pop=pop(SortOrder);
            
            % Update Worst Cost
            WorstCost=max(WorstCost,pop(end).Cost);
            
            % Truncation
            pop=pop(1:nPop);
            Costs=Costs(1:nPop);
            
            % Store Best Solution Ever Found
            BestSol=pop(1);
            
            % Store Best Cost Ever Found
            BestCost(10)=BestSol.Cost;
           rewardTemp = BestSol.Cost-(-1);
          if rewardTemp >0
                algRewards(10).rewards =1/rewardTemp;
            else
                algRewards(10).rewards =abs(rewardTemp);
          end
            
    end
    
    
    
    disp(['not Iter ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end
[~,index] = sortrows([algRewards.rewards].');
algRewards = algRewards(index(end:-1:1));
clear index
Algs = (1:nAlg);
tic
BestSolPerIter = pop(50);
for iter=1:MaxIt
    
    %lotCH = randi(Algs,1,1); % random by ranking
%      r=rand;
%      C=cumsum([algRewards.rewards]);
%      lotCH=find(r<=C,1,'first');


%------------------
% if iter>1
% if BestCostMaxiter(iter-1)>mean(BestCostMaxiter)
%      lotCH = lotchList(iter-1);
%      
% 
% else
%     randIdcs = randperm(length(Algs),1);
%     lotCH = Algs(randIdcs);
%     
% end
% else
%     randIdcs = randperm(length(Algs),1);
%     lotCH = Algs(randIdcs);
%     
% end
%-------------------------------
chooseRandomic = rand(1);
sumRewards=0;
for i=1:length(algRewards)
    sumRewards = sumRewards+algRewards(i).rewards;
end
rewardsAve=sumRewards/nAlg;

if iter<10
    lotCH=iter;
elseif chooseRandomic > RLAlpha 
    numi = randi([1,10]);
 if numi==1
        lotCH = algRewards(1).value;
    elseif numi==2
        lotCH = algRewards(2).value;
    elseif numi==3
        lotCH = algRewards(3).value;
    elseif numi==4
        lotCH = algRewards(4).value;
    elseif numi==5
        lotCH = algRewards(5).value;
    elseif numi==6
        lotCH = algRewards(6).value;
    elseif numi==7
        lotCH = algRewards(7).value;
    elseif numi==8
        lotCH = algRewards(8).value;
    elseif numi==9
        lotCH = algRewards(9).value;
    else
        lotCH = algRewards(10).value;
 end
 elseif chooseRandomic < RLAlpha
     r = rand;
     f=[algRewards.rewards];
     c = cumsum(f);
     numi=find(r<=c,1,'first'); % roul

% lotchListLength = length(lotchList);
% randiNum = randi([1,lotchListLength]);
% numi = lotchList(randiNum);

 
   if numi==1
         lotCH = algRewards(1).value;
    elseif numi==2
        lotCH = algRewards(2).value;
    elseif numi==3
        lotCH = algRewards(3).value;
    elseif numi==4
        lotCH = algRewards(4).value;
    elseif numi==5 
        lotCH = algRewards(5).value;
    elseif numi==6
        lotCH = algRewards(6).value;
    elseif numi==7
        lotCH = algRewards(7).value;
    elseif numi==8
        lotCH = algRewards(8).value;
    elseif numi==9
        lotCH = algRewards(9).value;
    else
        lotCH = algRewards(10).value;
  end  
end
lotchList = [lotchList,lotCH];
    
    %% -------------------------------------------------------------
    
    
    switch lotCH
  case 1
             
      newpop=pop;
      for i=1:nPop
          for k=1:nVar
              % Migration
              if rand<=lambda(i)
                  % Emmigration Probabilities
                  EP=bboMu;
                  EP(i)=0;
                  EP=EP/sum(EP);
                  
                  % Select Source Habitat
                  j=RouletteWheelSelection(EP);
                  
                  % Migration
                  newpop(i).Position(k)=pop(i).Position(k) ...
                      +bboAlpha*(pop(j).Position(k)-pop(i).Position(k));
                  
              end
              
              % Mutation
              if rand<=pMutation
                  newpop(i).Position(k)=newpop(i).Position(k)+bboSigma*randn;
              end
          end
          
          % Apply Lower and Upper Bound Limits
          newpop(i).Position = max(newpop(i).Position, VarMin);
          newpop(i).Position = min(newpop(i).Position, VarMax);
          
          % Evaluation
          newpop(i).Cost=CostFunction(newpop(i).Position);
      end
      
      % Sort New Population
      [~, SortOrder]=sort([newpop.Cost]);
      newpop=newpop(SortOrder);
      
      % Select Next Iteration Population
      pop=[pop(1:nKeep);newpop(1:bbonNew-1);BestSolPerIter];
      %pop(nPop) = BestSolPerIter;
      
      
      % Sort Population
      [~, SortOrder]=sort([pop.Cost]);
      pop=pop(SortOrder);
      
      % Update Best Solution Ever Found
      BestSol=pop(1);
      
      
      
      
      % Store Best Cost Ever Found
      BestCostMaxiter(iter)=BestSol.Cost;
      if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(1).rewards =1/rewardsTemp;
          else
              algRewards(1).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
           algRewards(1).rewards = algRewards(1).rewards-rewardsAve ;
           
      else
          algRewards(1).rewards = algRewards(1).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
      end
    
      
      
      case 2
         
          
         for ide=1:nPop
        
        x=pop(ide).Position;
        
        A=randperm(nPop);
        
        A(A==ide)=[];
        
        a=A(1);
        b=A(2);
        c=A(3);
        
        % Mutation
        %beta=unifrnd(beta_min,beta_max);
        beta=unifrnd(beta_min,beta_max,VarSize);
        y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);
        y = max(y, VarMin);
		y = min(y, VarMax);
		
        % Crossover
        z=zeros(size(x));
        j0=randi([1 numel(x)]);
        for j=1:numel(x)
            if j==j0 || rand<=pCR
                z(j)=y(j);
            else
                z(j)=x(j);
            end
        end
        
        newpop(it).Position=z;
        newpop(it).Cost=CostFunction(newpop(it).Position);
        
        if newpop(it).Cost<pop(ide).Cost
            pop(ide)=newpop(it);
            
            if pop(ide).Cost<BestSol.Cost
               BestSol=pop(ide);
            end
        end
        
    end
    if BestSolPerIter.Cost< BestSol.Cost
        BestSol = BestSolPerIter;
    end
    % Update Best Cost
    BestCostMaxiter(iter)=BestSol.Cost;
    if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(2).rewards =1/rewardsTemp;
          else
              algRewards(2).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(2).rewards = algRewards(2).rewards-rewardsAve ;
      else
          algRewards(2).rewards = algRewards(2).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
     end
  
          
      case 3
          
        temppop=pop;
        %repmat(empty_individual,nPop,1);
    for i=1:nPop
        %temppop(i).Cost = inf;
        for j=1:nPop
            if pop(j).Cost < pop(i).Cost
                rij=norm(pop(i).Position-pop(j).Position)/dmax;
                beta=beta0*exp(-ffGamma*rij^m);
                e=delta*unifrnd(-1,+1,VarSize);
                %e=delta*randn(VarSize);
                
                newpop(i).Position = pop(i).Position ...
                                + beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
                                + ffAlpha*e;
                
                newpop(i).Position=max(newpop(i).Position,VarMin);
                newpop(i).Position=min(newpop(i).Position,VarMax);
                
                newpop(i).Cost=CostFunction(newpop(i).Position);
                
                if newpop(i).Cost <= temppop(i).Cost
                    temppop(i) = newpop(i);
                    if temppop(i).Cost<=BestSol.Cost
                        BestSol=temppop(i);
                    end
                end
                
            end
        end
    end
    
    % Merge
    pop=[pop; temppop;BestSolPerIter];  %#ok
    
    % Sort
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    % Truncate
    pop=pop(1:nPop);
    
    % Store Best Cost Ever Found
    
    
    ffAlpha = ffAlpha*alpha_damp;
    
    % Store Best Cost Ever Found
    BestCostMaxiter(iter)=BestSol.Cost;
   if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(3).rewards =1/rewardsTemp;
          else
              algRewards(3).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(3).rewards = algRewards(3).rewards-rewardsAve ;
      else
          algRewards(3).rewards = algRewards(3).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
    end
    
      case 4
          
          
           % Initialize Array for New Harmonies
    NEW=pop;
    %repmat(empty_individual,nNew,1);
    
    % Create New Harmonies
    for k=1:nNew
        
        % Create New Harmony Position
        NEW(k).Position=unifrnd(VarMin,VarMax,VarSize);
        for j=1:nVar
            if rand<=HMCR
                % Use Harmony Memory
                i=randi([1 HMS]);
                NEW(k).Position(j)=pop(i).Position(j);
            end
            
            % Pitch Adjustment
            if rand<=PAR
                %DELTA=FW*unifrnd(-1,+1);    % Uniform
                DELTA=FW*randn();            % Gaussian (Normal) 
                NEW(k).Position(j)=NEW(k).Position(j)+DELTA;
            end
        
        end
        
        % Apply Variable Limits
        NEW(k).Position=max(NEW(k).Position,VarMin);
        NEW(k).Position=min(NEW(k).Position,VarMax);

        % Evaluation
        NEW(k).Cost=CostFunction(NEW(k).Position);
        
    end
    
    % Merge Harmony Memory and New Harmonies
    pop=[pop;NEW;BestSolPerIter]; %#ok
    
    % Sort Harmony Memory
     
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
   
    % Truncate Extra Harmonies
    pop=pop(1:HMS);
    
    % Update Best Solution Ever Found
    BestSol=pop(1);
    FW=FW*FW_damp;
    % Store Best Cost Ever Found
    BestCostMaxiter(iter)=BestSol.Cost;
    if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(4).rewards =1/rewardsTemp;
          else
              algRewards(4).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(4).rewards = algRewards(4).rewards-rewardsAve ;
      else
          algRewards(4).rewards = algRewards(4).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
    end
    
    
         
 case 5 
  for i=1:nPop
    
    % Initialize Position
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    pop(i).Velocity=zeros(VarSize);
    
    % Evaluation
    pop(i).Cost=CostFunction(pop(i).Position);
    
    % Update Personal Best
    pop(i).Best.Position=pop(i).Position;
    pop(i).Best.Cost=pop(i).Cost;
    
    % Update Global Best
         if pop(i).Best.Cost<BestSol.Cost
        
        GlobalBest=pop(i).Best;
        
         end
    
      end
          
           for i=1:nPop
        
        % Update Velocity
        pop(i).Velocity = w*pop(i).Velocity ...
            +c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-pop(i).Position);
        
        % Apply Velocity Limits
        pop(i).Velocity = max(pop(i).Velocity,VelMin);
        pop(i).Velocity = min(pop(i).Velocity,VelMax);
        
        % Update Position
        pop(i).Position = pop(i).Position + pop(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(pop(i).Position<VarMin | pop(i).Position>VarMax);
        pop(i).Velocity(IsOutside)=-pop(i).Velocity(IsOutside);
        
        % Apply Position Limits
        pop(i).Position = max(pop(i).Position,VarMin);
        pop(i).Position = min(pop(i).Position,VarMax);
        
        % Evaluation
        pop(i).Cost = CostFunction(pop(i).Position);
        
        % Update Personal Best
        if pop(i).Cost<pop(i).Best.Cost
            
            pop(i).Best.Position=pop(i).Position;
            pop(i).Best.Cost=pop(i).Cost;
            
            % Update Global Best
            if pop(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=pop(i).Best;
                
            end
            
        end
        
   end
           
           
    w=w*wdamp;
    if BestSolPerIter.Cost<GlobalBest.Cost
        GlobalBest = BestSolPerIter;
    end
    BestCostMaxiter(iter)=GlobalBest.Cost;
    if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(5).rewards =1/rewardsTemp;
          else
              algRewards(5).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(5).rewards = algRewards(5).rewards-rewardsAve ;
      else
          algRewards(5).rewards = algRewards(5).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
     end

          
      case 6
          
          
     for subit=1:MaxSubIt
        
        % Create and Evaluate New Solutions
        newpop=pop;
        %repmat(empty_individual,nPop,nMove);
        for i=1:nPop
            for j=1:nMove
                
                % Create Neighbor
                newpop(i,j).Position=SaMutate(pop(i).Position,saMu,saSigma,VarMin,VarMax);
                
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
    BestCostMaxiter(iter)=BestSol.Cost;
   if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(6).rewards =1/rewardsTemp;
          else
              algRewards(6).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(6).rewards = algRewards(6).rewards-rewardsAve ;
      else
          algRewards(6).rewards = algRewards(6).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
    end

    T=saAlpha*T;
    saSigma = 0.98*saSigma;
    
    
          
      case 7
         
          
          
           fla_params.BestSol = BestSol;

    % Initialize Memeplexes Array
    Memeplex = cell(nMemeplex, 1);
    
    % Form Memeplexes and Run FLA
    for j = 1:nMemeplex
        % Memeplex Formation
        Memeplex{j} = pop(I(j,:));
        
        % Run FLA
        Memeplex{j} = RunFLA(Memeplex{j}, fla_params);
        
        % Insert Updated Memeplex into Population
        pop(I(j,:)) = Memeplex{j};
    end
    
    % Sort Population
    pop = [pop;BestSolPerIter];
    pop = SortPopulation(pop);
    
    % Update Best Solution Ever Found
    BestSol = pop(1);
    
    % Store Best Cost Ever Found
   
    BestCostMaxiter(iter) = BestSol.Cost;
   
      if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(7).rewards =1/rewardsTemp;
          else
              algRewards(7).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(7).rewards = algRewards(7).rewards-rewardsAve ;
      else
          algRewards(7).rewards = algRewards(7).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
     end

    
    
      case 8
          
          
           % Calculate Population Mean
    Mean = 0;
    for i=1:nPop
        Mean = Mean + pop(i).Position;
    end
    Mean = Mean/nPop;
    
    % Select Teacher
    Teacher = pop(1);
    for i=2:nPop
        if pop(i).Cost < Teacher.Cost
            Teacher = pop(i);
        end
    end
    
    % Teacher Phase
    for i=1:nPop
        % Create Empty Solution
        newsol = empty_individual;
        
        % Teaching Factor
        TF = randi([1 2]);
        
        % Teaching (moving towards teacher)
        newsol.Position = pop(i).Position ...
            + rand(VarSize).*(Teacher.Position - TF*Mean);
        
        % Clipping
        newsol.Position = max(newsol.Position, VarMin);
        newsol.Position = min(newsol.Position, VarMax);
        
        % Evaluation
        newsol.Cost = CostFunction(newsol.Position);
        
        % Comparision
        if newsol.Cost<pop(i).Cost
            pop(i) = newsol;
            if pop(i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
        end
    end
    
    % Learner Phase
    for i=1:nPop
        
        A = 1:nPop;
        A(i)=[];
        j = A(randi(nPop-1));
        
        Step = pop(i).Position - pop(j).Position;
        if pop(j).Cost < pop(i).Cost
            Step = -Step;
        end
        
        % Create Empty Solution
        newsol = empty_individual;
        
        % Teaching (moving towards teacher)
        newsol.Position = pop(i).Position + rand(VarSize).*Step;
        
        % Clipping
        newsol.Position = max(newsol.Position, VarMin);
        newsol.Position = min(newsol.Position, VarMax);
        
        % Evaluation
        newsol.Cost = CostFunction(newsol.Position);
        
        % Comparision
        if newsol.Cost<pop(i).Cost
            pop(i) = newsol;
         
        end
    end
    pop(nPop) = BestSolPerIter;
    pop = [pop;BestSolPerIter];

    
    [~, SortOrder]=sort([pop.Cost]);
    pop = pop(SortOrder);
    

    % Competitive Exclusion (Delete Extra Members)
    if numel(pop)>nPop
        pop = pop(1:nPop);
    end
    
    % Store Best Solution Ever Found
    BestSol = pop(1);
    
    
    % Store Record for Current Iteration
   BestCostMaxiter(iter) = BestSol.Cost;
   if iter == 1
       rewardsTemp=BestSol.Cost-(-1);
       if rewardsTemp>0
           algRewards(8).rewards =1/rewardsTemp;
       else
           algRewards(8).rewards =abs(rewardsTemp);
       end
   elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
       %           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
       algRewards(8).rewards = algRewards(8).rewards-rewardsAve ;
   else
       algRewards(8).rewards = algRewards(8).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
   end

    
   
    
    
          
      case 9
          
          
          
          
    % Update Standard Deviation
    sigma = ((MaxIt - it)/(MaxIt - 1))^Exponent * (sigma_initial - sigma_final) + sigma_final;
    
    % Get Best and Worst Cost Values
    Costs = [pop.Cost];
    BestCost = min(Costs);
    WorstCost = max(Costs);
    
    % Initialize Offsprings Population
    newpop = pop;
    %[];
    
    % Reproduction
    for i = 1:numel(pop)
        
        ratio = (pop(i).Cost - WorstCost)/(BestCost - WorstCost);
        S = floor(Smin + (Smax - Smin)*ratio);
        
        for j = 1:S
            
            % Initialize Offspring
            newsol = empty_individual;
            
            % Generate Random Location
            newsol.Position = pop(i).Position + sigma * randn(VarSize);
            
            % Apply Lower/Upper Bounds
            newsol.Position = max(newsol.Position, VarMin);
            newsol.Position = min(newsol.Position, VarMax);
            
            % Evaluate Offsring
            newsol.Cost = CostFunction(newsol.Position);
            
            % Add Offpsring to the Population
            newpop = [newpop
                      newsol];  %#ok
            
        end
        
    end
    
    % Merge Populations
    pop = [pop;newpop;BestSolPerIter];
    
    % Sort Population
    
    [~, SortOrder]=sort([pop.Cost]);
    pop = pop(SortOrder);
    

    % Competitive Exclusion (Delete Extra Members)
    if numel(pop)>nPop
        pop = pop(1:nPop);
    end
    
    % Store Best Solution Ever Found
    BestSol = pop(1);
    
    
   BestCostMaxiter(iter) = BestSol.Cost;
   if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(9).rewards =1/rewardsTemp;
          else
              algRewards(9).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(9).rewards = algRewards(9).rewards-rewardsAve ;
      else
          algRewards(9).rewards = algRewards(9).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
    end
          
          
          
          
      case 10
          
          
             % Calculate Selection Probabilities
%     if UseRouletteWheelSelection
    P=exp(-gaBeta*Costs/WorstCost);
    P=P/sum(P);
%     end
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        
%         % Select Parents Indices
%         if UseRouletteWheelSelection
            i1=RouletteWheelSelection(P);
            i2=RouletteWheelSelection(P);
%         end
%         if UseTournamentSelection
%             i1=TournamentSelection(pop,TournamentSize);
%             i2=TournamentSelection(pop,TournamentSize);
%         end
%         if UseRandomSelection
%             i1=randi([1 nPop]);
%             i2=randi([1 nPop]);
%         end

        % Select Parents
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position,gaGamma,VarMin,VarMax);
        
        % Evaluate Offsprings
        popc(k,1).Cost=CostFunction(popc(k,1).Position);
        popc(k,2).Cost=CostFunction(popc(k,2).Position);
        
    end
    popc=popc(:);
    
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=Mutate(p.Position,gaMu,VarMin,VarMax);
        
        % Evaluate Mutant
        popm(k).Cost=CostFunction(popm(k).Position);
        
    end
    
    % Create Merged Population
    pop=[pop; popc;popm;BestSolPerIter]; %#ok
     
    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
    % Update Worst Cost
    WorstCost=max(WorstCost,pop(end).Cost);
    
    % Truncation
    pop=pop(1:nPop);
    
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost Ever Found
    BestCostMaxiter(iter)=BestSol.Cost;
    if iter == 1
          rewardsTemp=BestSol.Cost-(-1);
          if rewardsTemp>0
              algRewards(10).rewards =1/rewardsTemp;
          else
              algRewards(10).rewards =abs(rewardsTemp);  
          end
      elseif iter>11 && BestCostMaxiter(iter-1)==BestCostMaxiter(iter)
%           algRewards(1).rewards = algRewards(1).rewards-abs((BestCostMaxiter(iter-10)-BestCostMaxiter(iter-1)));
          algRewards(10).rewards = algRewards(10).rewards-rewardsAve ;
      else
          algRewards(10).rewards = algRewards(10).rewards+abs(BestCostMaxiter(iter)-BestCostMaxiter(iter-1));
    end
     
    
    
    
    
    
    
    
    
    end
  
%  if iter>1  
%  if BestCostMaxiter(iter)<BestCostMaxiter(iter-1)
%      Algs = [Algs,lotCH];
%      
%  end
%  
%  end
 RLAlpha = RLAlpha+(1/MaxIt); 
[~,index] = sortrows([algRewards.rewards].');
algRewards = algRewards(index(end:-1:1));
clear index
 BestSolPerIter = BestSol;  
 disp(['Iteration ' num2str(iter) ': Best Cost = ' num2str(BestCostMaxiter(iter) )]);
    

    
end

toc
disp(['Best Alg' algRewards(1).name]);


%%=========================================END=========================================
%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% Results

figure;
%plot(BestCost,'LineWidth',2);
semilogy(BestCostMaxiter,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
