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

% Divide the image into blocks
[numRows, numCols] = size(R);
numBlocksRows = floor(numRows / blockSize);
numBlocksCols = floor(numCols / blockSize);

blocks_R = mat2cell(R, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
%blocks_G = mat2cell(G_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);
%blocks_B = mat2cell(B_dct, blockSize*ones(1,numBlocksRows), blockSize*ones(1,numBlocksCols), 1);

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

% Compute the DCT coefficients of the text
 text_dct = dct(text_double);

% Normalize the DCT coefficients of the text to the range [-1, 1]
text_norm = 2*text_dct/length(text_dct) - 1;
disp(['The ASCII codes for the characters after DCT:  ' num2str(text_norm)]);
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
        % Convert red color channel into double precision
        R_double = im2double(blocks_R{i , j});

        % Compute the DCT coefficients of red color channel
        R_dct = dct2(R_double);
        block_R = R_dct;
        %block_G = blocks_G{i,j};
        %block_B = blocks_B{i,j};
        block_R_int = uint8(block_R); % Cast to integer type
        for k = 1:blockSize
            for l = 1:blockSize
                if keyIndex <= numel(numKey) && numKey(keyIndex) == k && numKey(keyIndex - 1) == l && watermarkIndex <= numel(binWatermark)
                    newBit = uint8 (binWatermark(watermarkIndex)) ;
                    pixel_R_new = bitset(block_R_int(k,l), 1, newBit); 
                    block_R_int(k,l) = pixel_R_new;
                    block_R = double(block_R_int);
                    blocks_R{i,j} = block_R;
                end
            end
        end
        block_R_idct = idct2(block_R);
        blocks_R{i,j} = block_R_idct;
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
R = cell2mat(blocks_R);
%G_dct = cell2mat(blocks_G);
%B_dct = cell2mat(blocks_B);

% Compute the inverse DCT of each color channel
%R_idct = idct2(R_dct);
%G_idct = idct2(G_dct);
%B_idct = idct2(B_dct);

% Combine the color channels into an RGB image
img_wm = cat(3, R, G, B);

% Remove the padding
img_wm = img_wm(1:end-padRows, 1:end-padCols, :);
%display the watermarked image 
figure;
imshow(img_wm);
imwrite(img_wm, 'output.jpg', 'jpg');

%extraction code 
% Initialize variables for the extracted watermark
extractedWatermark = '';
keyIndex = 2;

% Iterate over each block
for i = 1:numBlocksRows
    for j = 1:numBlocksCols
        block_R = blocks_R{i,j};
        
        for k = 1:blockSize
            for l = 1:blockSize
                if keyIndex <= numel(numKey) && numKey(keyIndex) == k && numKey(keyIndex - 1) == l
                    % Extract the least significant bit from the current pixel value in block_R
                    block_R_int = uint8(block_R); % Cast to integer type
                    watermarkBit = bitget(uint8(block_R_int(k, l)), 1);
                    % Append the watermark bit to the extractedWatermark string
                    extractedWatermark = strcat(extractedWatermark, num2str(watermarkBit));
                    % Update keyIndex
                    keyIndex = keyIndex + 2;
                end
            end
        end
    end
end

% Convert the binary string to the original watermark message
%extractedMessage = binaryVectorToASCII(extractedWatermark);

disp(extractedWatermark);
disp(size(extractedWatermark));

function text = bin2text(binaryStr)
    % Convert the binary string to a character array
    binaryArray = reshape(binaryStr, 8, [])';
    
    % Convert each row of the binary array to a decimal value
    decimalValues = bin2dec(binaryArray);
    
    % Convert the decimal values to characters
    text = char(decimalValues)';
end

function asciiMsg = binaryVectorToASCII(binaryMsg)
    % Convert the binary message to a matrix
    binaryMat = reshape(binaryMsg, 8, length(binaryMsg)/8)';
    % Convert each row of the binary matrix to decimal
    decimalMat = bi2de(binaryMat);
    % Convert the decimal matrix to ASCII
    asciiMsg = char(decimalMat)';
end


% function definition for modify_bit
function y = modify_bit(x, b)
    LSB = bitget(uint8(floor(x)), 1);
    if LSB == b
        y = x;
    else
        if b == 0
            y = x - 1;
        else
            y = x + 1;
        end
    end
end
