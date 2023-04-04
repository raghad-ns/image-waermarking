clc
clear all
close all 

% Define the key
key = 'hello world !!!!! hello world !!!!!!';

% Convert the key to binary data
binKey = dec2bin(uint8(key), 8);

% Reshape the binary data into a 2D array with 6 columns
binKey = reshape(binKey', [], 3);

% Convert the binary data to numeric data
numKey = bin2dec(binKey);
% disp (numKey);

% Load the image
img = imread('PeppersRGB.jpg');
% Block size
blockSize = 8; 

% pad image to make its size evenly divisible by the block size
padRows = blockSize - mod(size(img,1), blockSize);
padCols = blockSize - mod(size(img,2), blockSize);
img = padarray(img, [padRows padCols], 0, 'post');

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

% Divide the image into blocks
[numRows, numCols] = size(R_dct);
numBlocksRows = floor(numRows / blockSize);
numBlocksCols = floor(numCols / blockSize);

blocks_R = mat2cell(R_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
blocks_G = mat2cell(G_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
blocks_B = mat2cell(B_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);

% Load the text to be embedded as a watermark and convert it to a numeric vector of double precision
text = 'mousa';
text_double = double(text);

% Convert the watermark to binary data
binWatermark = dec2bin(uint8(text), 8);

% Reshape the binary data into array 
binWatermark = reshape(binWatermark', [], 1);


for i = 1 : numel(numKey)
    numKey(i) = numKey(i) + 1 ;
end

% Traverse through each block and each bit in the block, and embed the
% watermark
keyIndex = 2;
watermarkIndex = 1;
beforeWatermark = '';
%extractedWatermark = '';
% Main code starts here
for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        block_R = blocks_R{i,j};
        block_G = blocks_G{i,j};
        block_B = blocks_B{i,j};
        block_R_int = uint8(block_R); % Cast to integer type
        for k = 1:blockSize
            for l = 1:blockSize
                if keyIndex <= numel(numKey) && numKey(keyIndex) == k && numKey(keyIndex - 1) == l && watermarkIndex <= numel(binWatermark)
                    watermarkBit = bitget(uint8(block_R_int(k, l)), 1);
                    % Append the watermark bit to the extractedWatermark string
                    beforeWatermark = strcat(beforeWatermark, num2str(watermarkBit));
                    newBit = binWatermark(watermarkIndex) ;
                    newBit = bin2dec(newBit);
                    pixel_R_new = bitset(block_R_int(k,l), 1, newBit); 
                    block_R_int(k,l) = pixel_R_new;
                    block_R = double(block_R_int);
                    blocks_R{i,j} = block_R;
                    keyIndex = keyIndex + 2 ; 
                    watermarkIndex = watermarkIndex + 1 ;
                    %watermarkBit = bitget(uint8(block_R_int(k, l)), 1);
                    % Append the watermark bit to the extractedWatermark string
                    %extractedWatermark = strcat(extractedWatermark, num2str(watermarkBit));
                end
            end
        end
    end
end


for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        idx = (i-1)*numBlocksCols + j;
        subplot(numBlocksRows, numBlocksCols, idx);
        imshow(blocks_R{i,j});
    end
end

% Combine the blocks back into an image
R_dct = cell2mat(blocks_R);
G_dct = cell2mat(blocks_G);
B_dct = cell2mat(blocks_B);

% Compute the inverse DCT of each color channel
R_idct = idct2(R_dct);
G_idct = idct2(G_dct);
B_idct = idct2(B_dct);

% Combine the color channels into an RGB image
img_wm = cat(3, R_idct, G_idct, B_idct);

% Remove the padding
img_wm = img_wm(1:end-padRows, 1:end-padCols, :);
%display the watermarked image 

imwrite(img_wm, 'output.jpg', 'jpg');

% stand alone extraction 

% Define the key
key = 'hello world !!!!! hello world !!!!!!';

% Convert the key to binary data
binKey = dec2bin(uint8(key), 8);

% Reshape the binary data into a 2D array with 6 columns
binKey = reshape(binKey', [], 3);

% Convert the binary data to numeric data
numKey = bin2dec(binKey);
for i = 1 : numel(numKey)
    numKey(i) = numKey(i) + 1 ;
end
    
% Load the image
img = img_wm;
% Block size
blockSize = 8; 

% pad image to make its size evenly divisible by the block size
padRows = blockSize - mod(size(img,1), blockSize);
padCols = blockSize - mod(size(img,2), blockSize);
img = padarray(img, [padRows padCols], 0, 'post');

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

% Divide the image into blocks
[numRows, numCols] = size(R_dct);
numBlocksRows = floor(numRows / blockSize);
numBlocksCols = floor(numCols / blockSize);

blocks_R = mat2cell(R_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
blocks_G = mat2cell(G_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
blocks_B = mat2cell(B_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);

for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        idx = (i-1)*numBlocksCols + j;
        subplot(numBlocksRows, numBlocksCols, idx);
    end
end

% Initialize variables for the extracted watermark
extractedWatermark = '';
keyIndex = 2;

% Iterate over each block
for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        block_R = blocks_R{i,j};
        block_G = blocks_G{i,j};
        block_B = blocks_B{i,j};
        
        for k = 1:blockSize
            for l = 1:blockSize
                if keyIndex <= numel(numKey) && numKey(keyIndex) == k && numKey(keyIndex - 1) == l
                    % Extract the least significant bit from the current pixel value in block_R
                    block_R_int = uint8(block_R); % Cast to integer type
                    watermarkBit = bitget(block_R_int(k, l), 1);
                    % Append the watermark bit to the extractedWatermark string
                    extractedWatermark = strcat(extractedWatermark, num2str(watermarkBit));
                    % Update keyIndex
                    keyIndex = keyIndex + 2;
                end
            end
        end
    end
end

% convert binary watermark into text 
numChars = numel(extractedWatermark) / 8; % calculate number of characters in string
decimalVector = zeros(1,numChars); % preallocate decimal vector

% convert binary string to decimal vector
for i = 1:numChars
    startIdx = (i-1)*8+1;
    endIdx = i*8;
    decimalVector(i) = bin2dec(extractedWatermark(startIdx:endIdx));
end

% convert decimal vector to character vector
charVector = char(decimalVector);

%----------------------------------------------------------------
% Load the original and watermarked images
original = imread('PeppersRGB.jpg');
watermarked = imread('output.jpg');


%-------------------------------------------------------------------
% Add Gaussian noise to the watermarked image
noisy_img = imnoise(watermarked, 'gaussian', 0, 0.01);
figure
% Display the original, watermarked, and noisy watermarked images side by side
subplot(1, 3, 1);
imshow(original);
title('Original Image');

subplot(1, 3, 2);
imshow(watermarked);
title('Watermarked Image');

subplot(1, 3, 3);
imshow(noisy_img);
title('Noisy Watermarked Image');
%-----------------------------------------------------------------
% % Read in the two images
% I  = imread('PeppersRGB.jpg');
% 
% % Convert the images to grayscale (if they aren't already)
% if size(I,3) == 3
%     I = rgb2gray(I);
% end
% if size(noisy_img,3) == 3
%     noisy_img = rgb2gray(noisy_img);
% end
% 
% % Calculate the mean intensity of each image
% mean_I = mean(I(:));
% mean_J = mean(noisy_img(:));
% 
% % Calculate the standard deviation of each image
% std_I = std(I(:));
% std_J = std(noisy_img(:));
% 
% % Calculate the NC
% numerator = sum((I-mean_I).*(noisy_img-mean_J), 'all');
% denominator = sqrt(sum((I-mean_I).^2, 'all')*sum((noisy_img-mean_J).^2, 'all'));
% NC = numerator/denominator;
% 
% % Display the NC value
% fprintf('The NC between the two images is %f\n', NC);





