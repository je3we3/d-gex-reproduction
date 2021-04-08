%%
%setup
clc
clear all
close all
addpath('F:\BME_year1\Q3\DL\repro')
%%
%GEO data
data = parse_gctx('bgedv2_QNORM.gctx'); %load in data

%% extract matrices from data
dataMatrix = data.mat;
databaseGEO = data.mat;
[idx, clustered] = kmeans(transpose(databaseGEO),100);%perform kmeans clustering

clustered= clustered.clustered;
idx = idx.idx;
%%
dataT = transpose(dataMatrix);
%%
duplicates = [];
for i=1:100 %Loop over clusters
    currentClusterIdx=find(idx == i) ; %find all data in current cluster
    currentCluster = dataT(currentClusterIdx, :); %create current cluster matrix

    dist = pdist(currentCluster);%Pair wise distance calculation
    square = squareform(dist);%Convert output to matrix form
    n_dist = size(square,1);% find size of matrix square
    A = ones(n_dist);% create matrix full of ones with size of square
    underD= tril(A);%only lower triangle filled with ones to asure we do not find duplicates
    square_upper= square + 2*underD;% add 2 lower triangles to avoid duplicates
    [y, x] = find(square_upper<1);% Find all euclidean distances <1


    coord=[y x]; %Coordinate matrix duplicates


    duplicates= [duplicates ; currentClusterIdx(coord(:,2))];%save positions of duplicates
end

%%
udup=unique(duplicates);%Remove duplicate duplicates
nodup = dataT;
nodup(duplicates,:) = []; %Remove duplicates
%%
genes = data.rid;%create list with gene names
lmIdx=[];
for i= 1:size(maplm,1)
    lmIdx=[lmIdx find(ismember(genes,maplm(i,3)))]; %find lmgenes indices
end
%%
lmGenesData = nodup(:,lmIdx);%create matrix containing only lm genes data

%% create matrix containing only target and combined target genes
tgGenesData = [];
for i=1:size(maptg,1)%loop over map file
    currentGenes = split(maptg(i,3),','); %split the map file into loose genes
    currentID = find(ismember(data.rid,currentGenes)); %find all genes in current index
    currentMean = sum(nodup(:,currentID),2)./size(currentID,1); %mean current genes
    tgGenesData = [tgGenesData currentMean];%create matrix contain
end
%%
dataT=transpose([lmGenesData tgGenesData]);%transpose dataset to get to original form
qnorm = quantilenorm(dataT);%quantile normalization

%% create random 20% sample to be able to run it on colab
sampleIndex = randsample(1:111009,22202); %randomly select 20% of the indexes
samples = qnorm(:,sampleIndex);%create new sampled matrix
%%
save('F:\BME_year1\Q3\DL\repro\samples.mat', 'samples');
