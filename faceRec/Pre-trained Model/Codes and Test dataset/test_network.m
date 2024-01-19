function test_network(net, image)
    % Load and preprocess the image
    I = imread(image);
    
    % Resize the image to [224 224]
    G = imresize(I, [224, 224]);
    
    % Check the number of color channels
    if size(G, 3) == 1
        % If the image is grayscale, convert to RGB
        G = repmat(G, [1, 1, 3]);
    end
    
    % Classify the image using the pre-trained network
    [Label, Prob] = classify(net, G);

    % Display the image and the predicted label with its probability
    figure;
    imshow(G);
    title({char(Label), num2str(max(Prob), 2)});
end
