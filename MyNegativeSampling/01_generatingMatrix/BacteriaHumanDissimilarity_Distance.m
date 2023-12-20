% Dissimilarity distance for all-vs-all Bacteria proteins
% Part of: Dissimilarity-based Negative Sampling


%% STEP INPUT: Initialization
BLOUSUMno=30;
%%ResultFile=['BacteriaPathogenDistance' int2str(BLOUSUMno) '.mat'];
ResultFile=['BacteriaHumanDistance' int2str(BLOUSUMno) '.mat'];
%% STEP 1: Load Bacteria Protein Sequences
%%load BacteriaPathogen(Human).mat BacteriaProteinUnique
%%[~, SeqV]=fastaread('BacteriaPathogenFinal(Human)X.fasta');
load BacteriaHost(Human).mat BacteriaHumanProteinUnique
[~, SeqV]=fastaread('BacteriaHumanFinalX.fasta');

VpCount=length(SeqV);

%% STEP 2: Global Alignment
ScoringMatrix=['BLOSUM' num2str(BLOUSUMno)];
ScoreVps=zeros(VpCount,VpCount);
parfor r=1:VpCount
    fprintf("step2 doing %d \n ",r)
    for c=1:VpCount
        ScoreVps(r,c)= nwalign(SeqV{1,r},SeqV{1,c},'ScoringMatrix',ScoringMatrix); %#ok<PFBNS>
    end
end

%% STEP 3: Remove Outliers
%%fprintf("step3 doing \n")


%% STEP4: Normalize for Each Viral Protein (row-wise)
Weights=zeros(VpCount,VpCount);
for r=1:VpCount
    fprintf("step4 doing %d \n ",r)
    Row=ScoreVps(r,:);
    %Normalize on [0-1] scale, then reverse it (1-x)
    Weights(r,:)=1-((Row-min(Row))/(max(Row)-min(Row)));
end

%% STEP OUTPUT: Save
save(ResultFile,'ScoreVps','VpCount','BLOUSUMno','Weights')