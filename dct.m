% read the input image
img = imread('salad.jpg');

% convert the image to grayscale
img_gray = rgb2gray(img);

% convert the image to double precision
img_double = im2double(img_gray);

% compute the DCT of the image
img_dct = dct2(img_double);

% display the DCT coefficients
imshow(log(abs(img_dct)+1), []);
