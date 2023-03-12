% Load the RGB image and separate its color channels
img = imread('LenaRGB.jpg');
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
text = 'Raghad Dala';
text_double = double(text);

% Compute the DCT coefficients of the text
text_dct = dct(text_double);

% Normalize the DCT coefficients of the text to the range [-1, 1]
text_norm = 2*text_dct/length(text_dct) - 1;

% Embed the watermark into the DCT coefficients of the image
alpha = 0.1; % Watermark strength
text_norm_resized = imresize(text_norm, [size(R_dct,1), size(R_dct,2)]);
R_wm_dct = R_dct + alpha* text_norm_resized;
G_wm_dct = G_dct + alpha* text_norm_resized;
B_wm_dct = B_dct + alpha* text_norm_resized;

% Apply inverse DCT to obtain the watermarked color channels in the spatial domain
R_wm = idct2(R_wm_dct);
G_wm = idct2(G_wm_dct);
B_wm = idct2(B_wm_dct);

% Combine the watermarked color channels to obtain the watermarked RGB image
img_wm = cat(3, R_wm, G_wm, B_wm);

% Display the watermarked image
imshow(img_wm);
