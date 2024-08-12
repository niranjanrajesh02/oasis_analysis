% Clear workspace
clear;

% Set working directory
cd('C:/Niranjan/VisionLab/aesthetics/OASIS/OASIS-beauty/');

% Load data
depthDat = readtable('depthDat.csv');

nRatings = length([depthDat.beauty]);
r = zeros(100, 1); % Preallocate the r array
pi = zeros(100,1);

% Loop
for counter = 1:100
    % Shuffle data
    shuffledData = depthDat(randperm(nRatings),:);
    
    % Split data in half
    firstHalf = shuffledData(1:floor(nRatings/2),:);
    secondHalf = shuffledData(floor(nRatings/2)+1:end,:);
    
    % Calculate means for each item
    means1 = groupsummary(firstHalf, "item", "mean", "beauty");
    means2 = groupsummary(secondHalf, "item", "mean", "beauty");
    
    % Correlate the means
    [r(counter),p(counter)]  = nancorrcoef(means1{:,3}, means2{:,3});
    

end

% Summary stats
meanR = nanmean(r);
minR = min(r);
maxR = max(r);
maxP = max(p);
meanP = nanmean(p);

disp(['Mean correlation: ', num2str(meanR)]);
disp(['Mean p value: ', num2str(meanP)]);
