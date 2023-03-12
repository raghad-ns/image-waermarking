% Load the RGB image and separate its color channels
img = imread('LenaRGB.jpg');
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

% Convert each color channel into double precision
R_double = im2double(R);
G_double = im2double(G);
B_double = im2double(B);

% Convert the image to the frequency domain using DWT
[C,S] = wavedec2(R_double, 1, 'haar');

% Load the text to be embedded as a watermark and convert it to a numeric vector of double precision
text = 'Raghad Dala';
text_double = double(text);

% Convert the text to the frequency domain using DWT
[text_c, text_l] = wavedec(text_double, 1, 'haar');

% Resize the text coefficients to match the size of C
text_c_resized = imresize(text_c, size(C));

% Embed the watermark into the DWT coefficients of the image
alpha = 0.0001; % Watermark strength
C_wm = C + alpha * text_c_resized;

% Reconstruct the watermarked image
R_wm = waverec2(C_wm, S, 'haar');

% Combine the watermarked color channels to obtain the watermarked RGB image
img_wm = cat(3, R_wm, G_double, B_double);

% Display the watermarked image
%imshow(img_wm);

% Display the original and watermarked images side by side
figure;
subplot(1,2,1);
imshow(img);
title('Original Image');
subplot(1,2,2);
imshow(img_wm);
title('Watermarked Image');
