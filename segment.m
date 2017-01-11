% Number of images in the data set.
num_img = 188;

% How many segments in the corresponding image.
% Handcrafted number of segments, decided by experimenting.
% Read the number of segments for each image from a .mat file.
num_cuts = load('../numcuts.mat');

% Images in the data set.
images = cell(num_img, 1);
% Segment labels of the images after Ncut segmentation.
% segments = cell(num_img, 1);

for i = 113:num_img
    fprintf('Iteration %d', i);
    
    % Read the image.
    img = imread(strcat('../data/', num2str(i), '.jpg'));
    % Read its mask.
    mask = load(strcat('../data/', num2str(i), '.mat'));
    
    % Convert image to grayscale before applying Ncut.
    imgray = single(rgb2gray(img));
    
    % Perform the segmentation.
    [SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,W,imageEdges]= NcutImage(imgray, length(mask.masks));
    
    segments{i} = SegLabel;
end

% Save the segments to a .mat file.
% save('../segments.mat', segments);