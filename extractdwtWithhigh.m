clc
close all

% Load the host image and display it
rgbimage=imread('BaboonRGB.jpg');
figure;
imshow(rgbimage);
title('Original color image');

% Perform DWT on the host image and extract the low-low frequency coefficient
[h_LL,h_LH,h_HL,h_HH]=dwt2(rgbimage,'haar');
img=h_HH;

% Extract the red, green, and blue channels of the low-low frequency coefficient
red1=img(:,:,1);
green1=img(:,:,2);
blue1=img(:,:,3);

% Perform SVD on each channel
[U_imgr1,S_imgr1,V_imgr1]= svd(red1);
[U_imgg1,S_imgg1,V_imgg1]= svd(green1);
[U_imgb1,S_imgb1,V_imgb1]= svd(blue1);

% Load the watermark image and display it
rgbimage=imread('watermark.jpg');
figure;
imshow(rgbimage);
title('Watermark image');

% Perform DWT on the watermark image and extract the low-low frequency coefficient
[w_LL,w_LH,w_HL,w_HH]=dwt2(rgbimage,'haar');
img_wat=w_HH;

% Extract the red, green, and blue channels of the low-low frequency coefficient
red2=img_wat(:,:,1);
green2=img_wat(:,:,2);
blue2=img_wat(:,:,3);

% Perform SVD on each channel
[U_imgr2,S_imgr2,V_imgr2]= svd(red2);
[U_imgg2,S_imgg2,V_imgg2]= svd(green2);
[U_imgb2,S_imgb2,V_imgb2]= svd(blue2);

% Load the watermarked image and display it
key = 10
rgbimage=imread('watermarked.jpg');
figure;
imshow(rgbimage);
title('Watermarked image');

% Perform DWT on the watermarked image and extract the low-low frequency coefficient
[wm_LL,wm_LH,wm_HL,wm_HH]=dwt2(rgbimage,'haar');
img_w=wm_HH;

% Extract the red, green, and blue channels of the low-low frequency coefficient
red3=img_w(:,:,1);
green3=img_w(:,:,2);
blue3=img_w(:,:,3);

% Perform SVD on each channel
[U_imgr3,S_imgr3,V_imgr3]= svd(red3);
[U_imgg3,S_imgg3,V_imgg3]= svd(green3);
[U_imgb3,S_imgb3,V_imgb3]= svd(blue3);

% Extract the watermark from the host image
S_ewatr=(S_imgr3-S_imgr1)/(key*0.06);
S_ewatg=(S_imgg3-S_imgg1)/(key*0.06);
S_ewatb=(S_imgb3-S_imgb1)/(key*0.1);

ewatr = U_imgr2*S_ewatr*V_imgr2';
ewatg = U_imgg2*S_ewatg*V_imgg2';
ewatb = U_imgb2*S_ewatb*V_imgb2';

% Combine the extracted watermark channels into a single image
ewat=cat(3,ewatr,ewatg,ewatb);

% Set the low-low frequency coefficients of the watermarked image to the extracted watermark
newwatermark_HH=ewat;

% Reconstruct the watermarked image by performing inverse DWT on the watermarked low-low frequency coefficients along with the high frequency coefficients
rgb2=idwt2(newwatermark_HH,w_LH,w_HL,w_LL,'haar');

%display the Extracted image
figure;
imshow(uint8(rgb2));
imwrite(uint8(rgb2),'EWatermark.jpg');
title('Extracted Watermark');
%-------------------------------------------------------------------------------
% Load the original and watermarked images
original = imread('host.jpg');
watermarked = rgbimage;
watermark= imread('watermark.jpg');

% Convert the images to double precision for calculation
original = im2double(original);
watermarked = im2double(watermarked);

% Calculate the mean squared error (MSE) between the images
mse = mean((original(:)-watermarked(:)).^2);

% Calculate the maximum pixel value (assumes pixel values are in the range [0, 1])
max_pixel_value = 1;
if isinteger(original)
    max_pixel_value = double(intmax(class(original)));
end

% Calculate the PSNR in decibels (dB)
psnr = 10*log10((max_pixel_value^2)/mse);

% Display the PSNR value
disp(['PSNR value is: ', num2str(psnr), ' dB']);
% Display the MSE
fprintf('The Mean Squared Error between the images is: %0.2f\n', mse)


% Convert images to double precision and grayscale
img1 = im2double(rgb2gray(original));
img2 = im2double(rgb2gray(watermarked));

% Calculate SSIM
ssim_value = ssim(img1, img2);
% Display the result
fprintf('The SIM index between the two images is %f.\n', ssim_value);

%------------------------------------------------------------------------------
%for NC
fprintf('\n');
fprintf('\n');
I = im2double(rgb2gray(original));
W = im2double(rgb2gray(watermarked));

dct_I = dct2(I);
dct_W = dct2(W);

dct_I_high = dct_I(1:8:end, 1:8:end);
dct_W_high = dct_W(1:8:end, 1:8:end);

mean_I = mean(dct_I_high(:));
std_I = std(dct_I_high(:));

mean_W = mean(dct_W_high(:));
std_W = std(dct_W_high(:));


NC = sum((dct_I_high(:) - mean_I) .* (dct_W_high(:) - mean_W)) / (std_I * std_W * numel(dct_I_high));

disp(['NC = ', num2str(NC)]);

%-------------------------------------------------------------------------------
% Display the histograms for the original image, watermarked image, and watermark
figure;
subplot(3,1,1); imhist(original); title('Original Image');
subplot(3,1,2); imhist(watermark); title('Watermarked Image');
% subplot(3,1,3); imhist(watermarked); title('Watermarked Image');


