clc
clear all
close all 

% Load the original image
I = imread ('LenaRGB.jpg');
subplot (2,2,1), imshow(I),title('Original Image');

% Convert the image to double and apply 2D-DWT
IdR = im2double(I(:,:,1));
IdG = im2double(I(:,:,2));
IdB = im2double(I(:,:,3));
[LL1R,HL1R,LH1R,HH1R] = dwt2(IdR,'haar');
[LL1G,HL1G,LH1G,HH1G] = dwt2(IdG,'haar');
[LL1B,HL1B,LH1B,HH1B] = dwt2(IdB,'haar');

% Display the DWT of the original image
subplot(2,2,2),imshow(LL1R,[]),title('DWT for IO');

% Load the watermark image, convert it to grayscale and apply 2D-DWT
s = imread('sajeda-mug.jpg');
w = rgb2gray(s);
subplot(2,2,3),imshow(w), title('Watermark Image');
wd = im2double(w);
[LL1W,HL1W,LH1W,HH1W] = dwt2(wd,'haar');

% Display the DWT of the watermark image
subplot(2,2,4),imshow(LL1W,[]), title('DWT of WI');

% Embed the watermark in the original image using alpha=0.1
alpha = 0.1;
LL1wiR = LL1R + alpha * imresize(LL1W, size(LL1R));
LL1wiG = LL1G + alpha * imresize(LL1W, size(LL1G));
LL1wiB = LL1B + alpha * imresize(LL1W, size(LL1B));
WI = idwt2(LL1wiR,HL1R,LH1R,HH1R,'haar');
WI = cat(3, WI, idwt2(LL1wiG,HL1G,LH1G,HH1G,'haar'));
WI = cat(3, WI, idwt2(LL1wiB,HL1B,LH1B,HH1B,'haar'));

% Display the watermarked image
figure,
subplot (1,3,1), imshow(I),title('Original Image');
subplot(1,3,3),imshow(WI), title('Watermarked Image');

% Extract the watermark from the watermarked image
LL1ewR = (1/alpha) * (LL1wiR - LL1R);
LL1ewG = (1/alpha) * (LL1wiG - LL1G);
LL1ewB = (1/alpha) * (LL1wiB - LL1B);
EW = idwt2(LL1ewR,HL1R,LH1R,HH1R,'haar');
EW = cat(3, EW, idwt2(LL1ewG,HL1G,LH1G,HH1G,'haar'));
EW = cat(3, EW, idwt2(LL1ewB,HL1B,LH1B,HH1B,'haar'));

% Display the extracted watermark
subplot(1,3,2),imshow(EW),title('Extracted Watermark Image');

%HISTOGRAM
figure,
subplot (3,3,1), imhist(I), title('Histogram of OImge');
subplot (3,3,2), imhist(w), title('Histogram of W');
subplot (3,3,3), imhist(WI), title('Histogram of WD');

% Calculate PSNR and SSIM and MSE for the watermarked image
maxI = double(intmax(class(I)));
mse = mean(mean(mean((double(I) - double(WI)).^2)));
psnr = 10*log10(maxI^2/mse);
ssimval = ssim(im2double(I), im2double(WI));

fprintf('PSNR for the watermarked image: %f dB\n', psnr);
fprintf('SSIM for the watermarked image: %f\n', ssimval);
fprintf('MSE for the watermarked image: %f\n', mse);

