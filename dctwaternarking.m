clc
clear all
close all 

% Define the key
key = 'hello world!';

% Convert the key to binary data
binKey = dec2bin(uint8(key), 8);

% Reshape the binary data into a 2D array with 6 columns
binKey = reshape(binKey', [], 3);

% Convert the binary data to numeric data
numKey = bin2dec(binKey);
% disp (numKey);

% Load the image
img = imread('LenaRGB.jpg');
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

figure;
title('Host Image')
for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        idx = (i-1)*numBlocksCols + j;
        subplot(numBlocksRows, numBlocksCols, idx);
        imshow(blocks_R{i,j});
    end
end

% Load the text to be embedded as a watermark and convert it to a numeric vector of double precision
text = 'Raghad Dala sajeda';
text_double = double(text);

% Convert the watermark to binary data
binWatermark = dec2bin(uint8(key), 8);

% Reshape the binary data into array 
binWatermark = reshape(binWatermark', [], 1);

% Display the ASCII codes
disp(['The ASCII codes for the characters in the our name are:  ' num2str(text_double)]);

% Compute the DCT coefficients of the text
 text_dct = dct(text_double);

% Normalize the DCT coefficients of the text to the range [-1, 1]
text_norm = 2*text_dct/length(text_dct) - 1;
disp(['The ASCII codes for the characters after DCT:  ' num2str(text_norm)]);


% Traverse through each block and each bit in the block, and embed the
% watermark

keyIndex = 2;
watermarkIndex = 1 ;
for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        block_R = blocks_R{i,j};
        block_G = blocks_G{i,j};
        block_B = blocks_B{i,j};
        
        for k = 1:blockSize
            for l = 1:blockSize
                if keyIndex == k & keyIndex == (l + 1) & keyIndex <= size(numKey) & watermarkIndex <= size(binWatermark)
                    % Modify the bit at (k,l) in block_R
                    block_R(k,l) = modify_bit(block_R(k,l) , binWatermark(watermarkIndex));
                
                    % Modify the bit at (k,l) in block_G
                    block_G(k,l) = modify_bit(block_G(k,l) , binWatermark(watermarkIndex));
                
                    % Modify the bit at (k,l) in block_B
                    block_B(k,l) = modify_bit(block_B(k,l) , binWatermark(watermarkIndex));
                end
            end
        end
        
        % Assign the modified block back to the cell array
        blocks_R{i,j} = block_R;
        blocks_G{i,j} = block_G;
        blocks_B{i,j} = block_B;
    end
end

figure;
title ('Watermarked Image')
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
figure;
imshow(img_wm);
