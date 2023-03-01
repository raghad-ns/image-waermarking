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

% converting the watermark into frequancy domain to imbade it into the host
% image
% Step 1: Convert text into a sequence of numbers
text = 'Raghad Dala';
seq = double(text);

% Step 2: Apply DCT
dct_seq = dct(seq);

disp(seq);
% Step 3: Get magnitude and compress data range
mag_seq = abs(dct_seq);
log_mag_seq = log(mag_seq + 1);

% Plot the results
subplot(2,1,1);
plot(seq);
title('Text Signal');

subplot(2,1,2);
plot(log_mag_seq);
title('DCT Magnitude');
%imshow(log(abs(dct_seq)+1), []);

%just for testing
% Apply IDCT to retrieve text from DCT coefficients
recovered_seq = idct(dct_seq);

disp(recovered_seq);
% Convert sequence of numbers back to text
recovered_text = char(round(recovered_seq));

% Display the recovered text
disp(recovered_text);

