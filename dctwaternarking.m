clc
clear all
close all 

% Define a key
myString = 'Hello, world!';

% Convert the string into ASCII codes
asciiCodes = uint8(myString);

% Convert the ASCII codes into binary data
binaryData = reshape(dec2bin(asciiCodes, 8).', 1, []);

% Convert the binary data to numeric array
binaryDataNum = str2num(binaryData);

% Traverse through each bit of the binary data
for i = 1:numel(binaryDataNum)
    % Cast the input to an integer data type
    inputInt = int64(round(binaryDataNum(i)));
    
    % Get the value of the ith bit in the binary data
    bitValue = bitget(inputInt, 1:8);
    
    % Do something with the bit value
    disp(bitValue)
end


% Load the image
img = imread('PeppersRGB.jpg');
% Block size
blockSize = 8; 

% pad image to make its size evenly divisible by the block size
padRows = blockSize - mod(size(img,1), blockSize);
padCols = blockSize - mod(size(img,2), blockSize);
img = padarray(img, [padRows padCols], 0, 'post');

% divide image into blocks
nBlocksRows = size(img, 1) / blockSize;
nBlocksCols = size(img, 2) / blockSize;
blocks = mat2cell(img, blockSize*ones(1,nBlocksRows), blockSize*ones(1,nBlocksCols), 3);


% Display the blocks
figure;
for i = 1:nBlocksRows
    for j = 1:nBlocksCols
        idx = (i-1)*nBlocksCols + j;
        subplot(nBlocksRows, nBlocksCols, idx);
        imshow(blocks{i,j});
    end
end

% Load the RGB image and separate its color channels

R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

% Convert each color channel into double precision
R_double = im2double(R);
G_double = im2double(G);
B_double = im2double(B);

% Compute the DCT coefficients of each color channel
R_dct = dct2(R_double);
G_dct = dct2(G_double);
B_dct = dct2(B_double);

% Load the text to be embedded as a watermark and convert it to a numeric vector of double precision
text = 'Raghad Dala sajeda';
text_double = double(text);

% Display the ASCII codes
disp(['The ASCII codes for the characters in the our name are:  ' num2str(text_double)]);

% Compute the DCT coefficients of the text
 text_dct = dct(text_double);

% Normalize the DCT coefficients of the text to the range [-1, 1]
text_norm = 2*text_dct/length(text_dct) - 1;
disp(['The ASCII codes for the characters after DCT:  ' num2str(text_norm)]);

% Embed the watermark into the DCT coefficients of the image
alpha = 0.00027; % Watermark strength
text_norm_resized = imresize(text_norm, [size(R_dct,1), size(R_dct,2)]);
R_wm_dct = R_dct + alpha* text_norm_resized;
G_wm_dct = G_dct + alpha* text_norm_resized;
B_wm_dct = B_dct + alpha* text_norm_resized;
img_Dct = cat(3, R_wm_dct, G_wm_dct, B_wm_dct);
% Apply inverse DCT to obtain the watermarked color channels in the spatial domain
R_wm = idct2(R_wm_dct);
G_wm = idct2(G_wm_dct);
B_wm = idct2(B_wm_dct);

figure;
% Combine the watermarked color channels to obtain the watermarked RGB image
img_wm = cat(3, R_wm, G_wm, B_wm);

% Display the original image
subplot (2,2,1), imshow(img),title('Original Image');

% Display the DCT of the original image
subplot(2,2,2),imshow(log(abs(img_Dct)),[]),title('DCT for IO');

% Display the watermarked image
subplot (2,2,3), imshow(img_wm),title('WaterMarked Image ');

% Compute the MSE & PSNR between the two images
img = im2double(img);
img_wm = im2double(img_wm);

psnr_value = psnr(img, img_wm);
mse = immse(img, img_wm);

fprintf('MSE between original and watermarked image: %f\n', mse);
fprintf('PSNR between original and watermarked image: %f\n', psnr_value);
