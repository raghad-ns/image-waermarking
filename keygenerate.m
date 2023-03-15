% Define the struct
myStruct.name = "John Doe";
myStruct.age = 35;
myStruct.isMarried = true;

% Serialize the struct
binaryData = serializeStruct(myStruct);

% Write the binary data to a file
filename = "C:\Users\smart\Desktop\crypto\image-watermarking\key.bin.txt";
fid = fopen(filename, "wb");
if fid == -1
    error("Error: Could not open file %s", filename);
end
count = fwrite(fid, binaryData, "uint8");
if count ~= numel(binaryData)
    error("Error: Could not write all bytes to file %s", filename);
end
fclose(fid);

% Deserialize the binary data into a struct
newStruct = deserializeStruct(binaryData);

% Print the deserialized struct
disp(newStruct);

function binaryData = serializeStruct(structData)
% Serialize a struct into binary data
binaryData = getByteStreamFromArray(structData);
end

function structData = deserializeStruct(binaryData)
% Deserialize binary data into a struct
structData = getArrayFromByteStream(binaryData);
end
