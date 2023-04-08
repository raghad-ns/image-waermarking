# image-watermarking

files description : 
- dctwatermarking file 
  in this file we did the following steps : 
  1- load image & convert it to frequency domain using dct 
  2- embed the watermark into dct image using specific key 
  3- extract the watermark from watermarked image
  4- calculate (PSNR , MSE , SIM) 
  5- display histogram for original & watermarked image
  6- apply attack by adding Gaussian noise of power 0.02 to the image 
  7- calculate NC between watermarked image with & without noise
  8- trying to extract the watermark after adding noise
  
- dctWatermarkingHigh file 
  in this file we did the same things as dctwatermarking file 
  the only difference is that it adds & extracts the watermark in/from high frequency components

- embededDWT file 
  in this file we did the following steps : 
  1- load image & convert it to frequency domain using dwt 
  2- embed the watermark image into dwt image using specific key 

- extractDWT file
  1- extract the watermark image from watermarked image
  2- calculate (PSNR , MSE , SIM) 
  3-  display histogram for original & watermarked image

attackDWT :
  1- apply attack by adding noise of power 0.02 to the image 
  2- trying to extract the watermark after adding noise
  
- embededDWTWithHigh & extractdwtwithhigh file 
  in this file we did the same things as  file embededDWT& extractDWT  
  the only difference is that it adds & extracts the watermark in/from high frequency components
  
  
