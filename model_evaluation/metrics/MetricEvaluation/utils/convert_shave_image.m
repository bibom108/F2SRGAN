function shaved = convert_shave_image(input_image,shave_width)

% Converting to y channel only
if size(input_image, 3) == 3
        image_ychannel = rgb2ycbcr(input_image);
        image_ychannel = image_ychannel(:,:,1);
else
        image_ychannel = input_image;
end

% Shaving image
shaved = image_ychannel(1+shave_width:end-shave_width,...
        1+shave_width:end-shave_width);
    
end