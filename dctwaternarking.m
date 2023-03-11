% Load the image
img = imread('LenaRGB.jpg');

% Convert the image into grayscale
gray_img = rgb2gray(img);

% convert the image to double precision
img_double = im2double(gray_img);

% Apply 2D DCT to the grayscale image
dct_data = dct2(img_double);
imshow (dct_data);

% Convert the text to a binary sequence using ASCII codes
text = 'Raghad Dala';
binary_text = dec2bin(text, 8);
binary_text = binary_text(:)' - '0';

% Generate a PRBS with the same length as the binary sequence of the text
prbs = randi([0 1], 1, length(binary_text));

% Modulate the binary sequence of the text with the PRBS using XOR operation
spread_spectrum = xor(binary_text, prbs);

% Compute the absolute values of the DCT coefficients
abs_dct_data = abs(dct_data);

% Apply quantization to the absolute values
alpha = 0.01; % Adjust alpha to control the watermark strength
Q = quantiz(abs_dct_data(:), alpha);

% Reshape the quantized values to the same size as the DCT coefficients
Q = reshape(Q, size(abs_dct_data));

% Embed the spread spectrum signal into the frequency domain of the image using QIM
[dct_rows, dct_cols] = size(dct_data);
for i = 1:dct_rows
    for j = 1:dct_cols
        if Q(i, j) == 1 || Q(i, j) == -1
            dct_data(i, j) = dct_data(i, j) + alpha * sign(abs(dct_data(i, j))) * spread_spectrum(mod(i+j, length(spread_spectrum))+1);
        end
    end
end

imshow (dct_data);
% Apply inverse 2D DCT to the watermarked image
watermarked_dct = idct2(dct_data);

% Save the watermarked image
%imwrite(watermarked_dct, 'watermarked.jpg');
% Apply inverse DCT to obtain the watermarked image in spatial domain
watermarked_image_idct = idct2(watermarked_dct);

% Invert the conversion of image to double precision
img_gray = img_double * 255;
img_gray = uint8(img_gray);

% Invert the conversion of image to grayscale
img_rgb = ind2rgb(img_gray, gray(256));
imshow (img)