% ! plot layerwise correlation between activations and beauty
clear; clc; close all;

% load the beauty ratings
load('./analysis/OASIS_obj_scene.mat') % loads oasis_data as data
% take subset of data table where category == 'objects'
obj_idx = ismember(data.Category, 'Object');
obj_beauty_ratings = data.beauty_mean(obj_idx);
scene_idx = ismember(data.Category, 'Scene');
scene_beauty_ratings = data.beauty_mean(scene_idx);

clear data;
num_objects = length(obj_beauty_ratings);
num_scenes = length(scene_beauty_ratings);

train_category = 'Scene';
test_category = 'Object';

arch = 'vgg16';
train_acts_path = ['./analysis/activations/', train_category, '/', arch, '_pca_layers/'];
test_acts_path = ['./analysis/activations/', test_category, '/', arch, '_pca_layers/'];

if strcmp(train_category, 'Object')
    train_beauty_ratings = obj_beauty_ratings;
    test_beauty_ratings = scene_beauty_ratings;
    num_train = num_objects;
    num_test = num_scenes;
else
    train_beauty_ratings = scene_beauty_ratings;
    test_beauty_ratings = obj_beauty_ratings;
    num_train = num_scenes;
    num_test = num_objects;
end
clear obj_beauty_ratings scene_beauty_ratings;

n_layers = length(dir(fullfile(train_acts_path, '*.mat'))); % assumes same number of layers for both categories
n_folds = 10;

layerwise_corr = zeros(n_layers, n_folds);
components = zeros(n_layers, 1);

n_comp_thresh = 10;

for layer_i=1:n_layers
    layer_name = strcat('layer_', num2str(layer_i));
    
    % load train category activations
    load(fullfile(train_acts_path, strcat(layer_name, '.mat')));
    X_train = layer_representations;
    Y_train = train_beauty_ratings;
    clear layer_representations
    
    % load test category activations
    load(fullfile(test_acts_path, strcat(layer_name, '.mat')));
    X_test = layer_representations;
    Y_test = test_beauty_ratings;
    clear layer_representations
    
    min_pcs = min(size(X_train, 2), size(X_test, 2));
    if ~isnan(n_comp_thresh)
        min_pcs = min(min_pcs, n_comp_thresh);
    end
    
    X_train = [X_train(:, 1:min_pcs) ones(num_train, 1)];
    X_test = [X_test(:, 1:min_pcs) ones(num_test, 1)];
    
    corr_list = zeros(n_folds, 1);
    
    components(layer_i) = min_pcs;
    
    % 10 fold cross validation
    for fold_i = 1:n_folds
        %     train_idx = randperm(num_images, round(0.9*num_images));
        %     test_idx = setdiff(1:num_images, train_idx);
        b = regress(Y_train, X_train);
        pred = X_test * b;
        corr_list(fold_i) = corr(Y_test, pred);
    end
    
    layerwise_corr(layer_i,:) = corr_list;
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
subtitle(['Trained on: ', train_category, ' Tested on: ', test_category,  ', PCs - ', n_pcs_str]);


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

