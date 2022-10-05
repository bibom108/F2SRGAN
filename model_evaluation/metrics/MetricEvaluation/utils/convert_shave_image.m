function image_ychannel = convert_shave_image(input_image,shave_width)

% Converting to y channel only
if size(input_image, 3) == 3
        image_ychannel = rgb2ycbcr(input_image);
        image_ychannel = image_ychannel(:,:,1);
else
        image_ychannel = input_image;
end

end