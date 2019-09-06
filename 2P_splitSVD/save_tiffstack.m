function []=save_tiffstack(stack,title)
%This function saves grayscale tiff stack of score_maps or glomMasks

if ndims(stack)==3
imwrite(mat2gray(stack(:,:,1)),strcat(title,'.tiff'),'tiff')
for i=2:size(stack,3)
    imwrite(mat2gray(stack(:,:,i)),strcat(title,'.tiff'),'tiff','WriteMode','append')
end
elseif ndims(stack)==4
imwrite(mat2gray(stack(:,:,:,1)),strcat(title,'.tiff'),'tiff')
for i=2:size(stack,4)
    imwrite(mat2gray(stack(:,:,:,i)),strcat(title,'.tiff'),'tiff','WriteMode','append')
end    
end