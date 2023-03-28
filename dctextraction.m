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
for i = 1 : numel(numKey)
    numKey(i) = numKey(i) + 1 ;
end

% Load the image
img = imread('output.jpg');
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
                    watermarkBit = bitget(uint8(block_R(k, l)), 1);
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

