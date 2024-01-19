%% FOllowing Toolboxes are required: Parallel Computing Toolbox, Deep Learning Toolbox,  Image Processing Toolbox, Statistics and Machine Learning Toolbox

gpu = gpuDevice(1); % Select the first GPU (modify the index as needed)


Dataset= imageDatastore('My_dataset', 'IncludeSubfolders',true, 'LabelSource', 'foldernames');
[Training_Dataset, Validation_Dataset]= splitEachLabel(Dataset, 7.0);

net = googlenet;

analyzeNetwork(net);

Input_Layer_Size= net.Layers(1).InputSize;

Layer_Graph= layerGraph(net);

Featur_Learner= net.Layers(142);
Output_Classifier= net.Layers(144);

Number_Of_Classes= numel(categories(Training_Dataset.Labels));

New_Feature_Learner= fullyConnectedLayer(Number_Of_Classes, ...
    'Name', 'Facial Feature Learner', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);


New_Classifier_Layer= classificationLayer('Name', 'Face Classifier');

Layer_Graph= replaceLayer(Layer_Graph, Featur_Learner.Name, New_Feature_Learner);

Layer_Graph= replaceLayer(Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);

analyzeNetwork(Layer_Graph);

Pixel_Range= [-30 30];
Scale_Range= [0.9 1.1];

Image_Augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', Pixel_Range, ...
    'RandYTranslation', Pixel_Range, ...
    'RandXScale', Scale_Range, ...
    'RandYScale', Scale_Range, ...
    'RandRotation',[0 360]);

% Specify 'ColorPreprocessing' to ensure consistent number of channels
Augmented_Training_Image = augmentedImageDatastore(Input_Layer_Size(1:2), Training_Dataset, ...
    'DataAugmentation', Image_Augmenter, 'ColorPreprocessing', 'gray2rgb');

Augmented_Validation_Image = augmentedImageDatastore(Input_Layer_Size(1:2), Validation_Dataset, 'ColorPreprocessing', 'gray2rgb');

Size_of_Minibatch = 5;
Validation_Frequency = floor(numel(Augmented_Training_Image.Files) / Size_of_Minibatch);

Training_Options = trainingOptions('sgdm', ...   % stochastic gradient descent
    'ExecutionEnvironment', 'gpu',...
    'MiniBatchSize', 5, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Augmented_Validation_Image, ...
    'ValidationFrequency', Validation_Frequency, ...
    'Verbose', false, ...
    'MaxEpochs', 90, ...
    'Plots', 'training-progress');

net = trainNetwork(Augmented_Training_Image, Layer_Graph, Training_Options);

save('my_trained_model.mat', 'net', 'Training_Options');
%%

test_network(net, 'image1.jpg')
test_network(net, 'image2.jpg')
test_network(net, 'image3.jpg')
test_network(net, 'image4.jpg')
test_network(net, 'image5.jpg')
test_network(net, 'image6.jpg')
test_network(net, 'image7.jpg')
test_network(net, 'image8.jpg')
test_network(net, 'image9.jpg')
test_network(net, 'image10.jpg')
test_network(net, 'image11.jpg')
test_network(net, 'image12.jpg')
test_network(net, 'image13.jpg')
test_network(net, 'image14.jpg')
test_network(net, 'image15.jpg')
test_network(net, 'image16.jpg')
test_network(net, 'image17.jpg')
test_network(net, 'image18.jpg')
test_network(net, 'image19.jpg')
test_network(net, 'image20.jpg')
test_network(net, 'image21.jpg')