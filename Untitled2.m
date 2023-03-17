% set up variables
key = 'secretkey';
img = imread('LenaRGB.jpg');
img_size = size(img);
block_size = [8, 8];
num_blocks = floor(img_size ./ block_size);

% convert key to binary
key_bin = de2bi(uint8(key), 'left-msb');
key_bin = key_bin(:);

% create blocks
blocks = mat2cell(img, block_size(1) * ones(1, num_blocks(1)), block_size(2) * ones(1, num_blocks(2)), 3);
for i = 1:numel(blocks)
    blocks{i}(1,1,:) = bitset(blocks{i}(1,1,:), 1, key_bin(mod(i-1,length(key_bin))+1));
end

% reconstruct image from blocks
new_img = cell2mat(blocks);

% display original and modified images
subplot(1,2,1);
imshow(img);
title('Original Image');
subplot(1,2,2);
imshow(new_img);
title('Modified Image');
