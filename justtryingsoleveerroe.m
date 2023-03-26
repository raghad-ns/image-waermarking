




















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


blockSize = 4;
[numRows, numCols] = size(R_dwt);
numBlocksRows = floor(numRows/blockSize);
numBlocksCols = floor(numCols/blockSize);

% Pad R_dwt with zeros if necessary to make its size an even multiple of blockSize
R_dwt_padded = padarray(R_dwt, [blockSize-mod(numRows,blockSize), blockSize-mod(numCols,blockSize)], 'post');

% Reshape padded R_dwt into a 3D array
R_dwt_reshaped = reshape(R_dwt_padded, blockSize, numBlocksRows, blockSize, numBlocksCols);

% Convert R_dwt_reshaped to a cell array of 2D blocks
blocks_R = mat2cell(R_dwt_reshaped, blockSize, blockSize, ones(1,numBlocksRows*numBlocksCols));

% Apply wavelet thresholding to each block in blocks_R
for i = 1:numBlocksRows*numBlocksCols
    blocks_R{i} = wthresh(blocks_R{i}, 's', threshold);
end

% Reshape blocks_R back into a 4D array
R_dwt_thresholded = reshape(blocks_R, blockSize, numBlocksRows, blockSize, numBlocksCols);
R_dwt_thresholded = permute(R_dwt_thresholded, [2 4 1 3]);
R_dwt_thresholded = reshape(R_dwt_thresholded, numBlocksRows*blockSize, numBlocksCols*blockSize);

















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

% Compute the DWT coefficients of each color channel
R_dwt = dwt2(R_double, 'haar');
G_dwt = dwt2(G_double, 'haar');
B_dwt = dwt2(B_double, 'haar');

% Divide the image into blocks
% [numRows, numCols] = size(R_dwt);
% numBlocksRows = floor(numRows / blockSize);
% numBlocksCols = floor(numCols / blockSize);




blockSize = [3 3];
[rows, cols] = size(Image);
numBlocksRows = ceil(rows / blockSize(1));
numBlocksCols = ceil(cols / blockSize(2));
D1 = repmat(blockSize, numBlocksRows, numBlocksCols);
 


[numRows, numCols] = size(R_dwt_thresholded);
numRows = numRows + padRows;
numCols = numCols + padCols;

numBlocksRows = numRows / blockSize;
numBlocksCols = numCols / blockSize;


D1 = repmat(blockSize, 1, numBlocksRows);
D2 = repmat(blockSize, 1, numBlocksCols);






% 
% blocks_R = mat2cell(R_dwt, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
% blocks_G = mat2cell(G_dwt, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
% blocks_B = mat2cell(B_dwt, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);

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
text = 'Raghad';
text_double = double(text);

% Convert the watermark to binary data
binWatermark = dec2bin(uint8(text), 8);


% Reshape the binary data into array 
binWatermark = reshape(binWatermark', [], 1);

% Display the ASCII codes
disp(['The ASCII codes for the characters in the our name are:  ' num2str(text_double)]);

% Compute the DWT coefficients of the text
 text_dwt = dwt(text_double, 'haar');

% Normalize the DWT coefficients of the text to the range [-1, 1]
text_norm = text_dwt / max(abs(text_dwt));
disp(['The ASCII codes for the characters after DWT:  ' num2str(text_norm)]);
for i = 1 : numel(numKey)
    numKey(i) = numKey(i) + 1 ;
end

% Traverse through each block and each bit in the block, and embed the
% watermark
keyIndex = 2;
watermarkIndex = 1;

% Main code starts here
for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        block_R = blocks_R{i,j};
        block_G = blocks_G{i,j};
        block_B = blocks_B{i,j};
        
        for k = 1:blockSize
            for l = 1:blockSize
                if keyIndex <= numel(numKey) && numKey(keyIndex) == k && numKey(keyIndex - 1) == l && watermarkIndex <= numel(binWatermark)
                    %disp(block_R(k , l));
                    watermarkBit =uint8(binWatermark(watermarkIndex));
                    % Modify the bit at (k,l) in block_R
                    %block_R_int = uint8(block_R); % Cast to integer type
                    %block_R_bitset = bitset(reshape(block_R_int, 1, 64), 1, watermarkBit); % Apply bitset
                    %block_R_int_modified = typecast(uint8(block_R_bitset), 'uint8'); % Convert back to integer type

                    %block_R(k,l) = bitset(reshape(uint8(block_R), 1, 64), 1, watermarkBit);

                    % Modify the bit at (k,l) in block_G
                    block_G(k,l) = bitset(uint8(block_G(k,l)), 1, watermarkBit);
                
                    % Modify the bit at (k,l) in block_B
                    %block_B(k,l) = bitset(uint8(block_B(k,l)), 1, watermarkBit);
                    
            
     % Finish modifying the bit at (k,l) in block_R
                %block_R(k,l) = typecast(block_R_int_modified, 'double'); % Convert back to double type
                %blocks_R{i,j} = block_R; % Store the modified block back into the cell array

                watermarkIndex = watermarkIndex + 1; % Increment the index into the watermark
            end
            keyIndex = keyIndex + 1; % Increment the index into the key
        end
    end
    
    % Replace the old block with the modified block in each channel
    blocks_R{i,j} = block_R;
    blocks_G{i,j} = block_G;
    blocks_B{i,j} = block_B;
    end


   end

% Convert each color channel back into a 2D matrix
R_embedded = cell2mat(blocks_R);
G_embedded = cell2mat(blocks_G);
B_embedded = cell2mat(blocks_B);

% Reconstruct the RGB image with the embedded watermark
img_embedded = cat(3, R_embedded, G_embedded, B_embedded);
img_embedded = uint8(idwt2(R_embedded, G_embedded, B_embedded, 'haar'));

% Save the watermarked image
imwrite(img_embedded, 'PeppersRGB_watermarked.png');

% Display the watermarked image
figure;
subplot(1,2,1);
imshow(img);
title('Host Image');
subplot(1,2,2);
imshow(img_embedded);
title('Watermarked Image'); 
    
    
    
    
    
% figure;
% title ('Watermarked Image')
% for i = 1:numBlocksRows
%     for j = 1:numBlocksCols
%         idx = (i-1)*numBlocksCols + j;
%         subplot(numBlocksRows, numBlocksCols, idx);
%         imshow(blocks_R{i,j});
%     end
% end
% 
% % Combine the blocks back into an image
% R_dct = cell2mat(blocks_R);
% G_dct = cell2mat(blocks_G);
% B_dct = cell2mat(blocks_B);
% 
% % Compute the inverse DCT of each color channel
% R_idct = idct2(R_dct);
% G_idct = idct2(G_dct);
% B_idct = idct2(B_dct);
% 
% % Combine the color channels into an RGB image
% img_wm = cat(3, R_idct, G_idct, B_idct);
% 
% % Remove the padding
% img_wm = img_wm(1:end-padRows, 1:end-padCols, :);
% %display the watermarked image 
% figure;
% imshow(img_wm);
% imwrite(img_wm, 'output2.jpg', 'jpg');
% 
% 
% % function definition for modify_bit
% function y = modify_bit(x, b)
%     LSB = bitget(uint8(floor(x)), 1);
%     if LSB == b
%         y = x;
%     else
%         if b == 0
%             y = x - 1;
%         else
%             y = x + 1;
%         end
%     end
% end
