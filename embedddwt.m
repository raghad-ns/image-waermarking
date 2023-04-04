% Clear the command window and close all figures
clc
close all

% Load the host image
rgbimage = imread('host.jpg');

% Display the original image
figure;
imshow(rgbimage);
title('Original color image');

% Apply 2D Haar Wavelet Transform to the host image and extract the 
% low-low, low-high, high-low, and high-high frequency coefficients
[h_LL, h_LH, h_HL, h_HH] = dwt2(rgbimage, 'haar');
img = h_LL;

% Extract the red, green, and blue channels of the low-low frequency coefficients
red1 = img(:, :, 1);
green1 = img(:, :, 2);
blue1 = img(:, :, 3);

% Apply Singular Value Decomposition (SVD) to each channel of the low-low frequency coefficients
[U_imgr1, S_imgr1, V_imgr1] = svd(red1);
[U_imgg1, S_imgg1, V_imgg1] = svd(green1);
[U_imgb1, S_imgb1, V_imgb1] = svd(blue1);

% Load the watermark image
rgbimage = imread('watermark.jpg');

% Display the watermark image
figure;
imshow(rgbimage);
title('Watermark image');

% Apply 2D Haar Wavelet Transform to the watermark image and extract the 
% low-low, low-high, high-low, and high-high frequency coefficients
[w_LL, w_LH, w_HL, w_HH] = dwt2(rgbimage, 'haar');
img_wat = w_LL;

% Extract the red, green, and blue channels of the low-low frequency coefficients
red2 = img_wat(:, :, 1);
green2 = img_wat(:, :, 2);
blue2 = img_wat(:, :, 3);

% Apply Singular Value Decomposition (SVD) to each channel of the watermark image
[U_imgr2, S_imgr2, V_imgr2] = svd(red2);
[U_imgg2, S_imgg2, V_imgg2] = svd(green2);
[U_imgb2, S_imgb2, V_imgb2] = svd(blue2);

% Define a key for watermarking
key = 10;

% Watermarking:
% Add a percentage (0.10) of the singular values of the watermark image 
% to the singular values of the host image in each channel
S_wimgr = S_imgr1 + (0.10 * S_imgr2 * key);
S_wimgg = S_imgg1 + (0.10 * S_imgg2 * key);
S_wimgb = S_imgb1 + (0.10 * S_imgb2 * key);

% Reconstruct each channel of the watermarked image by multiplying the 
% corresponding U, S, and V matrices obtained from the SVD
wimgr = U_imgr1 * S_wimgr * V_imgr1';
wimgg = U_imgg1 * S_wimgg * V_imgg1';
wimgb = U_imgb1 * S_wimgb * V_imgb1';

% Combine the red, green, and blue channels of the watermarked image 
% to form the final watermarked image
wimg = cat(3, wimgr, wimgg, wimgb);
newhost_LL = wimg;
newhost_LH = wimg;
newhost_HL = wimg;
newhost_HH = wimg;

% Reconstruct the watermarked image by performing inverse DWT on the 
% watermarked low-low frequency coefficients along with the original 
% low-high, high-low, and high-high frequency coefficients
watermarked_rgb2=idwt2(newhost_LH,h_LL,h_HL,h_HH,'haar');

watermarked = cat(3, h_LL, newhost_LH, h_HL, h_HH);

imwrite(uint8(watermarked_rgb2),'Watermarked.jpg');
figure;
imshow(uint8(watermarked_rgb2));
title('Watermarked Image');