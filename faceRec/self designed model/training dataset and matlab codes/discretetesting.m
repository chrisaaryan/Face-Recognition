%Program to recognize faces discretely
load('C:\Users\DELL\Desktop\PR PROJECT\self designed model\training dataset and matlab codes\wpcadb.mat', 'n','m','M','N','Ppca','T');

%wpcadb.mat loads followings in the workspace :
%Number of total traiing images [n], Image size [M,N], Mean Image [m]
%Reduced Eigen vectors transformation matrix [Ppca]
%Transformed dataset matrix [T]

[filename,pathname]= uigetfile('*.*', 'Select the Input Face Image');
filewithpath= strcat(pathname,filename);
img= imread(filewithpath);
imgo=img; %Copying image for display
img= rgb2gray(img);
img= imresize(img, [M,N]);
level= graythresh(img);
Ibin= imbinarize(img,level);

%Finding Discete wavelet transform
dwtmode('per','nodisp');
[cA,cH,cV,cD]= dwt2(double(Ibin), 'db10');
wc= [cA,cH;cV,cD]; %Wavelet coefficients

%Finding standard deviation of wavelet coefficients
stdcol= std(wc);
wcc= (wc');
stdrow= std(wcc);
fvstd= [stdcol stdrow]; %Feature vector using STD
fvpca= (fvstd-m)*Ppca; %Projecting fv to PCA space

figure;

% Display input image
subplot(1, 3, 1);
imshow(imgo);
title('Input Test Image');

% Display binarized image after feature extraction
subplot(1, 3, 2);
imshow(Ibin);
title('Binarized Image');


distarray= zeros(n,1); %Initialize difference array

for i=1:n
    distarray(i)= sum(abs(T(i,:)-fvpca)); %Finding L1 distance between the input feature vector and each training feature vector stored in the matrix T.
end

%-----------Displaying best match-----------------
[result,indx]= sort(distarray); %sorting the array
resultimg1= imread(sprintf('%d.jpg', indx(1)));

subplot(133); imshow(resultimg1); title('Best matched Image')