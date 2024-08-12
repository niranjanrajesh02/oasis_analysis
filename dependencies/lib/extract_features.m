% ! Function in progress. Args = (images, arch)

function nlayers = extract_features(images, arch, save_path)
% OUT: nlayers: number of layers in the pretrained network
% SAVED: layerwise activations for all images in save_path
% IN: images: cell array of images
%     arch: architecture of the pretrained network ('vgg16')
%     save_path: path to save the extracted features


% arch = 'vgg16';
% arch has to be 'vgg16' or 'ViT_b'
if(strcmp(arch,'vgg16'))
    [net,~] = imagePretrainedNetwork('vgg16');
elseif(strcmp(arch,'resnet18'))
    [net,~] = imagePretrainedNetwork('resnet18');
else
    return
end

nimages=length(images);
img_size = net.Layers(1).InputSize(1:2);

layers = net.Layers;
% only extract activations from layer that has 'Convolution' or 'Fully Connected' in its name
layers = layers(arrayfun(@(x) isa(x, 'nnet.cnn.layer.Convolution2DLayer') || isa(x, 'nnet.cnn.layer.FullyConnectedLayer'), layers));
nlayers = length(layers);

% ! normalising images
% TODO: normalise for specific dataset
avg_img = net.Layers(1).Mean;
if size(avg_img,1)==1
    old_avg_img = avg_img;
    for chan=1:3
        avg_img(:,:,chan) = old_avg_img(chan);
    end
end
img_sd = net.Layers(1).StandardDeviation;
if isempty(img_sd)
    img_sd = [1,1,1];
end


% ! MAIN LOOP
for layer_i=1:nlayers
    layerwise_features = cell(nimages, 1);
    layer_name = layers(layer_i).Name;
    for img_i=1:nimages
        img = single(images{img_i});
        if size(img,3)==1
            img = repmat(img,1,1,3);
        end
        img = imresize(img, img_size);
        % normalisation
        img = img - avg_img;
        for chan=1:3,img(:,:,chan) = img(:,:,chan) / img_sd(chan);end
        % extract acts
        features = predict(net, img, Outputs=layer_name);
        layerwise_features{img_i} = features;
    end
    disp("Layer " + layer_name + " (" + layer_i + "/" + nlayers + ") done.");
    save_file = [save_path, 'layer_', num2str(layer_i), '.mat'];
    save(save_file, 'layerwise_features','-v7.3');
    clear layerwise_features;
end












