% ! plot layerwise correlation between activations and beauty
clear; clc; close all;

% load the beauty ratings
load('./analysis/OASIS_data.mat') % loads oasis_data as data
beauty_ratings = data.beauty_mean;
num_images = length(beauty_ratings);
clear data;

arch = 'vgg16';
layer_path = ['./analysis/activations/', arch, '_pca_layers/'];
n_layers = length(dir(fullfile(layer_path, '*.mat')));
n_folds = 10;
layerwise_corr = zeros(n_layers, n_folds);
components = zeros(n_layers, 1);

for i=1:n_layers
    layer_name = strcat('layer_', num2str(i));
    load(fullfile(layer_path, strcat(layer_name, '.mat')));
    
    X = [layer_representations ones(num_images)];
    Y = beauty_ratings;
    corr_list = zeros(n_folds, 1);
    components(i) = size(layer_representations, 2);
    
    % 10 fold cross validation
    for j = 1:n_folds
        train_idx = randperm(num_images, round(0.9*num_images));
        test_idx = setdiff(1:num_images, train_idx);
        b = regress(Y(train_idx), X(train_idx, :));
        pred = X(test_idx, :) * b;
        corr_list(j) = corr(Y(test_idx), pred);
    end
    
    layerwise_corr(i,:) = corr_list;
    display("Layer " + layer_name + " done.");
    
end

avg_layerwise_corr = mean(layerwise_corr, 2);
sd = std(layerwise_corr, 0, 2);

% check if all components are the same
if isscalar(unique(components))
    n_pcs_str = num2str(components(1));
else
    n_pcs_str = ['min:', num2str(min(components)), ' max:', num2str(max(components))];
end

% plot the correlation values
figure;

plot(avg_layerwise_corr, 'LineWidth', 2);
hold on;
errorbar(1:n_layers, avg_layerwise_corr, sd, 'LineStyle', 'none', 'LineWidth', 0.1);
xlim([1 n_layers+2]);
% line at y=0
hold on;
plot([0 n_layers+2], [0 0], '--k');
xlabel('Layer');
ylabel('Correlation');
title(['Laywerwise Correlation between activations and beauty ratings for ', arch]);
subtitle(['PCs - ', n_pcs_str]);


% ! for layer x, plotting predicted and actual beauty ratings

% layer_name = 'layer_37';
% load(fullfile('./analysis/activations/pca_layers', strcat(layer_name, '.mat')));

% % load the beauty ratings
% load('./analysis/OASIS_data.mat') % loads oasis_data as data
% beauty_ratings = data.beauty_mean;

% num_images = length(beauty_ratings);
% X = [layer_representations ones(num_images)];
% Y = beauty_ratings;


% train_idx = randperm(num_images, round(0.8*num_images));
% test_idx = setdiff(1:num_images, train_idx);
% b = regress(Y(train_idx), X(train_idx, :));
% pred = X(test_idx, :) * b;

% figure;
% scatter(Y(test_idx), pred, 15, 'red', 'filled');
% % plot training points diff colour
% hold on;
% scatter(Y(train_idx), X(train_idx, :) * b, 10, 'black', 'filled');

% xlabel('Actual Beauty Ratings');
% ylabel('Predicted Beauty Ratings');
% title(['Actual vs Predicted Beauty Ratings for Layer', ' r = ', num2str(corr(Y(test_idx),pred), 3)]);
% % unity line
% hold on;
% plot([0 10], [0 10], 'black');

