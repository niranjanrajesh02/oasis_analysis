clear; clc; close all;
addpath('./analysis/dependencies/lib/');
% ! set params and load vars
load('./analysis/OASIS_obj_scene.mat') % loads as data

% take subset of data table where category == 'objects'
category = 'Object'; % either 'Object' or 'Scene'
idx = ismember(data.Category, category);
data = data(idx, :);
img_set = data.img;
clear data;

arch = 'vgg16';
act_save_path = ['./analysis/activations/', category, '/', arch, '_layers/'];

% ! extract activations for each image in img_set
% check if folder is non empty
if length(dir(fullfile(act_save_path, '*.mat'))) == 0
    n_layers = extract_features(img_set, arch, act_save_path);
    disp("Activations extracted for " + n_layers + " layers.");
else
    disp("Activations already extracted.");
    n_layers = length(dir(fullfile(act_save_path, '*.mat')));
end

% ! PCA on each activation layer to reduce dimensionality
% load activations
pca_save_path = ['./analysis/activations/', category, '/', arch, '_pca_layers/'];

n_components = 200;
for i = 1:n_layers% iterate through layers
    layer_name = strcat('layer_', num2str(i));
    load(fullfile(act_save_path, strcat(layer_name, '.mat')));
    layer = cellfun(@(x) x(:), layerwise_features, 'UniformOutput', false); % flatten activations of each image
    layer = [layer{:}]';
    
    [proj, npc, ve, pcs] = pcaproject(layer, 0.95);
    
    % print npc, ve, pcs
    fprintf("n_PCs that explain 0.95 variance for layer %s: %d \n", layer_name, npc);
    if npc < n_components
        n_components = npc;
    end
    layer_representations = proj(:, 1:n_components);
    save(fullfile(pca_save_path, strcat(layer_name, '.mat')), 'layer_representations');
    
    % print
    fprintf('PCA reps saved for %s\n', layer_name);
end