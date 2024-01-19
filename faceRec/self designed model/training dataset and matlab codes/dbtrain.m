%program to implement face recognition using wavelet features and PCA
%Why do we perform dwt?
% the Discrete Wavelet Transform (DWT) is applied to the binary images of faces.
% which is used as a feature extraction technique before performing Principal Component Analysis
n= 800; %No. of training images

L= 70; %No of dominant Eigen values selected to be used in PCA for dimensionality reduction

M= 200; N= 180; %Required image dimensions

X= zeros(n, (M+N)); %initializing data set matrix
for count= 1:n
    I= imread(sprintf('%d.jpg', count)); %reading images
    I= rgb2gray(I); %RGB to grayscale
    I= imresize(I, [M,N]); %resize all the images to specified M*N
    level= graythresh(I); %threshold value comuted using Otsu's method
    Ibin= imbinarize(I,level); %Getting binary image
    %Finding discrete Wavelet Transform
    dwtmode('per', 'nodisp');
    [cA,cH,cV,cD]=dwt2(double(Ibin), 'db10'); %Ddb10= Daubechies wavelet with 10 coeff. 
    wc= [cA,cH;cV,cD]; %Wavelet coefficients arranged
    stdcol= std(wc); %Colwise standard dev
    wcc= (wc');
    stdrow= std(wcc); %Rowwise standard dev
    fvstd= [stdcol stdrow]; %std along columns and rows are concatenated and stored in a single feature vector
    X(count,:)= fvstd; %Saving all feature vectors
end

%Projecting all the feature vectors to PCA space
m= mean(X); %Mean of all feature vectors

for i= 1:n
    X(i,:)= X(i,:)-m; %Subtracting mean from each feature vector for normalization of data
end

%PCA calculation

Q= (X'*X)/(n-1); %Finding covariance matrix

[Evecm,Evalm]= eig(Q); %Getting eigen values and eigen vectors of matrix Q which provides the direction and magnitude of Princial components
Eval= diag(Evalm); %Getting eigen values
[Evalsorted,Index]= sort(Eval, 'descend'); %Sorting eigen values
Evecsorted= Evecm(:,Index); %Getting corresponding eigen vectors
Ppca= Evecsorted(:,1:L); %Reduced transformation matrix Ppca
T= X*Ppca; %Projecting each feature vector to PCA space

save('C:\Users\DELL\Desktop\PR PROJECT\self designed model\training dataset and matlab codes\wpcadb.mat', 'n','m','M','N','Ppca','T');
