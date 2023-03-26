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

% Load the watermarked image
img_wm = imread('output.jpg');

% Decompose the watermarked image using DWT
[LL3, HL3, LH3, HH3] = dwt2(img_wm, 'haar');
[LL2, HL2, LH2, HH2] = dwt2(LL3, 'haar');
[LL1, HL1, LH1, HH1] = dwt2(LL2, 'haar');

% Extract the watermark from the approximation coefficients LL1
numRows = size(LL1, 1);
numCols = size(LL1, 2);
extractedBits = '';
keyIndex = 2;
watermarkIndex = 1;
for i = 1:numRows
    for j = 1:numCols
        if keyIndex <= numel(numKey) && numKey(keyIndex) == i && numKey(keyIndex - 1) == j
            % Extract the bit at (i,j) from LL1
            extractedBits = [extractedBits, num2str(bitget(uint8(LL1(i,j)),1))];

            %update keyIndex & watermarkIndex 
            keyIndex = keyIndex + 2 ;
            watermarkIndex = watermarkIndex + 1 ;
        end
    end
end

% Convert the extracted bits to the original watermark
disp(extractedBits);
extractedWatermark = bin2text(extractedBits);
disp(extractedWatermark);

function text = bin2text(binaryStr)
    % Add zero padding to make the length of the binary string divisible by 8
    numPaddingBits = mod(length(binaryStr), 8);
    if numPaddingBits ~= 0
        binaryStr = [binaryStr, repmat('0', 1, 8 - numPaddingBits)];
    end
    
    % Convert the binary string to a character array
    binaryArray = reshape(binaryStr, 8, [])';
    
    % Convert each row of the binary array to a decimal value
    decimalValues = bin2dec(binaryArray);
    
    % Convert the decimal values to characters
    text = char(decimalValues)';
end

