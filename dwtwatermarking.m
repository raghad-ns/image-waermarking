% Load the image and the text
img = imread('LenaRGB.jpg');
text = 'raghad dala';

% Convert the image to grayscale
img_gray = rgb2gray(img);

% Convert the image to the frequency domain using DWT
[C,S] = wavedec2(img_gray, 1, 'haar');
[H1,V1,D1] = detcoef2('all',C,S,1);
A1 = appcoef2(C,S,'haar',1);
imshow (A1);

% Convert the text to a vector of ASCII codes and apply DWT
text_arr = double(text);
text_coeffs = dwt(text_arr,'haar');

% Choose the coefficients to modify
CD1 = D1(:);
alpha = 0.1;
for i = 1:length(text_coeffs)
    if text_coeffs(i) > 0
        CD1(i) = CD1(i) + alpha * abs(CD1(i));
    else
        CD1(i) = CD1(i) - alpha * abs(CD1(i));
    end
end
D1_modified = reshape(CD1,size(D1));

% Transform the modified coefficients back to the spatial domain
C_new = [A1(:)', H1(:)', V1(:)', D1_modified(:)'];
watermarked_img = waverec2(C_new,S,'haar');
%imshow(watermarked_img)

% Save the watermarked image
%imwrite(watermarked_img, 'watermarked_image.png');
