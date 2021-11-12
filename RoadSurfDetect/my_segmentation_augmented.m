rng(777)

imgDir_train = fullfile('dataset_augmented\train\Image');
imgDir_val = fullfile('dataset_augmented\val\Image');
imgtrain = imageDatastore(imgDir_train);
imgval = imageDatastore(imgDir_val);

clsspixelvalue = [1 2 3 4];
class_name = ["none","dry","wet", "snow"];
labDir_train = fullfile('dataset_augmented\train\Label');
labDir_val = fullfile('dataset_augmented\val\Label');
labtrain = pixelLabelDatastore(labDir_train, class_name, clsspixelvalue);
labval = pixelLabelDatastore(labDir_val, class_name, clsspixelvalue);
table_label = labtrain.countEachLabel();
freq = table_label.PixelCount./table_label.ImagePixelCount;
medain_freg = median(freq);
med_freq_bal = medain_freg./freq;

%%
imageSize = [256 256 3];
numClasses = numel(class_name);
dlv3p = deeplabv3plusLayers(imageSize, numClasses,'resnet18');

pix_classify_layer = pixelClassificationLayer('Name','labels',...
    'Classes',table_label.Name,'ClassWeights',med_freq_bal);
dlv3p = replaceLayer(dlv3p,"classification",pix_classify_layer);

val_dataset = pixelLabelImageDatastore(imgval,labval);

options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ... 
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',val_dataset,...
    'MaxEpochs',20, ... % 30 
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', '', ...
    'VerboseFrequency',2,...
    'ValidationPatience', inf, ... % 50
    'ValidationFrequency', 200, ...
    'Plots','training-progress');

augmenter_opt = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandRotation', [-10, 10]);
train_dataset = pixelLabelImageDatastore(imgtrain,labtrain, ...
    'DataAugmentation',augmenter_opt);
[net, info] = trainNetwork(train_dataset,dlv3p, options);

nn_augmented = net;
save nn_augmented nn_augmented
%%
% load nn_augmented
% clsspixelvalue = [1 2 3 4];
% class_name = ["none","dry","wet", "snow"];
imgDir_test = fullfile('dataset_augmented\test\Image');
imgtest = imageDatastore(imgDir_test);
labDir_test = fullfile('dataset_augmented\test\Label');
labtest = pixelLabelDatastore(labDir_test, class_name, clsspixelvalue);

for i=1:length(imgtest.Files)
    img = readimage(imgtest,i); 
    sur_detect = semanticseg(img,nn_augmented); %dry,wet,snow    
    detect_none = sur_detect == 'none';
    detect_dry = sur_detect == 'dry';
    detect_wet = sur_detect == 'wet';
    detect_snow = sur_detect == 'snow';
    detectResult = uint8(detect_none + 2* detect_dry ...
        + 3* detect_wet + 4* detect_snow);
    imwrite(detectResult, strcat('dataset_augmented\test\Result\', ...
        num2str(i, '%04d'), '.png'));    
end
%%
resultDir_test = fullfile('dataset_augmented\test\Result');
resulttest = pixelLabelDatastore(resultDir_test,...
    class_name, clsspixelvalue);
metrics_tst_aug = evaluateSemanticSegmentation(resulttest,labtest);

save('metrics\metrics_tst_aug','metrics_tst_aug')
%%
% load nn_augmented
% clsspixelvalue = [1 2 3 4];
% class_name = ["none","dry","wet", "snow"];
imgDir_new = fullfile('dataset_new\image');
imgnew = imageDatastore(imgDir_new);
labDir_new = fullfile('dataset_new\label');
labnew = pixelLabelDatastore(labDir_new, class_name, clsspixelvalue);

for i=1:length(imgnew.Files)
    img = readimage(imgnew,i); 
    sur_detect = semanticseg(img,nn_augmented); %dry,wet,snow    
    detect_none = sur_detect == 'none';
    detect_dry = sur_detect == 'dry';
    detect_wet = sur_detect == 'wet';
    detect_snow = sur_detect == 'snow';
    detectResult = uint8(detect_none + 2* detect_dry ...
        + 3* detect_wet + 4* detect_snow);
    imwrite(detectResult, strcat('dataset_new\Result_aug\', ...
        num2str(i, '%04d'), '.png'));    
end
%%
resultDir_new = fullfile('dataset_new\Result_aug');
resultnew = pixelLabelDatastore(resultDir_new,...
    class_name, clsspixelvalue);
metrics_new_aug = evaluateSemanticSegmentation(resultnew,labnew);

save('metrics\metrics_new_aug','metrics_new_aug')