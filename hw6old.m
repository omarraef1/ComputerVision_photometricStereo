function hw6()

% Lecture 13 slides 7 thru 21


close all;
format compact;
%datacursormode on;
%figure, imshow('4-2.tiff')
%figure, imshow('4-3.tiff')
%figure, imshow('4-4.tiff')
%figure, imshow('4-5.tiff')
%figure, imshow('4-6.tiff')
%figure, imshow('4-7.tiff')

%im1 = imread('4-1.tiff');
%whos('im1')
% 1. Compute normals
% light illumination direction of camera:
%       0, 0, 1
% Canonical view of normal direction:
%       0, 0, 1 
% w/ Pixel value of:
%       250, 250, 250

% Step 1: Bi(x,y) = Vi(dot)n(x,y)

%lightDir(1,:)

%for i = 1:7
    
%end

%_______STARTING OVER________

% find p's and q's

lightDir = load('light_directions.txt');
lightDir
%lightDir(1,:)
%{
p1 = lightDir(1,1)/-lightDir(1,3);
q1 = lightDir(1,2)/-lightDir(1,3);
p1
q1
[xn1, yn1, zn1] = normals(p1,q1);
xn1
yn1
zn1
%}



img1 = imread('4-1.tiff');
%mask = 256*ones(400,400,'uint8');
img2 = imread('4-2.tiff');
img3 = imread('4-3.tiff');
img4 = imread('4-4.tiff');
img5 = imread('4-5.tiff');
img6 = imread('4-6.tiff');
img7 = imread('4-7.tiff');
mask1 = im2bw(img1);
mask1 = (mask1>0);
mask2 = im2bw(img2);
mask2 = (mask2>0);
mask3 = im2bw(img3);
mask3 = (mask3>0);
mask4 = im2bw(img4);
mask4 = (mask4>0);
mask5 = im2bw(img5);
mask5 = (mask5>0);
mask6 = im2bw(img6);
mask6 = (mask6>0);
mask7 = im2bw(img7);
mask8 = (mask7>0);
mask9 = imfuse(mask1, mask2);
mask10 = imfuse(mask9, mask3);
mask11 = imfuse(mask10, mask4);
mask12 = imfuse(mask11, mask5);
mask13 = imfuse(mask12, mask6);
mask = imfuse(mask13, mask7);
imshow(mask)
% test
%[p2, q2] = getPQ(img1);
%[xn2, yn2, zn2] = normals(p2,q2);
% test end
imgArr = {img1; img2; img3; img4; img5; img6; img7};
Normsies = computeNorms(imgArr, lightDir, mask);
%Normsies
whos('Normsies')
%whos('img1')
size(imgArr,1)
size(lightDir,1)
whos('imgArr')
whos('lightDir')
shaepe = drawNorm(Normsies, 4);
%depthM = computeDepths(Normsies, mask);
%figure, imshow(uint8(depthM));
nuDepth = computeDepthDraft(Normsies, mask, 400, 400);

end

function depth = computeDepthDraft(N, mask, h, w)
len=h*w;
A=sparse(len*2,len);
B=zeros(len*2,1);
for y=1:h-1
    for x=1:w-1
        i=(y-1)*w+x;
        A(2*i-1,i)=-N(y,x,3);
        A(2*i-1,i+1)=N(y,x,3);
        B(2*i-1)=N(y,x,1);

        A(2*i,i)=-N(y,x,3);
        A(2*i,i+w)=N(y,x,3);
        B(2*i)=N(y,x,2);
    end
end
newrow=A(1:2*(len-1),1:len)\B(1:2*(len-1),1);
depth=reshape(newrow,w,h)';
% depth=depth-min(min(depth));
X=[1:w];
Y=[1:h];
Z=depth;
 
% z = reshape(z,r,c);
% z = z-min(min(z));


figure,
surfl(X,Y,Z)
shading interp
colormap bone
view([200 100 3500]);

figure,
surfl(X,Y,Z)
shading interp
colormap bone
view([190 90]);

end

function depth = computeDepths(N, mask)
[im_h, im_w, ~] = size(N);
whos('mask')

[obj_h, obj_w] = find(mask);
obj_h = 400;
obj_w = 400;
no_pix = size(obj_h, 1);
full2obj = zeros(im_h, im_w);
for i = 1:size(obj_h, 1)
    full2obj(obj_h(i), obj_w(i)) = i;
end

M = sparse(2*no_pix, no_pix);
u = sparse(2*no_pix, 1);

ignoredRows = [];

for i = 1:no_pix-1
    h = obj_h(i);
    w = obj_w(i);
    n_x = N(h, w, 1);
    n_y = N(h, w, 2);
    n_z = N(h, w, 3);
    
    row_i = (i-1)*2+1;
    if mask(h+1, w)
        i_vertN = full2obj(h+1, w);
        u(row_i) = -n_y;
        M(row_i, i) = -n_z;
        M(row_i, i_vertN) = n_z;
    elseif mask(h-1, w)
        i_vertN = full2obj(h-1, w);
        u(row_i) = -n_y;
        M(row_i, i) = -n_z;
        M(row_i, i_vertN) = n_z;
    else
        ignoredRows = [ignoredRows; row_i];
    end
    
    row_i = (i-1)*2+2;
    if mask(h, w+1)
        i_horizN = full2obj(h, w+1);
        u(row_i) = -n_x;
        M(row_i, i) = -n_z;
        M(row_i, i_horizN) = n_z;
    elseif mask(h, w-1)
        i_horizN = full2obj(h, w-1);
        u(row_i) = n_x;
        M(row_i, i) = -n_z;
        M(row_i, i_horizN) = n_z;
    else
        ignoredRows = [ignoredRows; row_i];
    end
end

M(ignoredRows,:) = [];
u(ignoredRows, :) = [];

z = (M.'*M)\(M.'*u);

z = full(z);
outlier = abs(zscore(z))>10;
z_min = min(z(~outlier));
z_max = max(z(~outlier));

depth = double(mask);
for i = 1:no_pix
    h = obj_h(i);
    w = obj_w(i);
    depth(h,w) = (z(i)-z_min)/(z_max-z_min)*255;
end
    

end

function shape = drawNorm(N, step)
[im_h, im_w, ~] = size(N);

[X, Y] = meshgrid(1:step:im_w, im_h:-step:1);
U = N(1:step:im_h, 1:step:im_w, 1);
V = N(1:step:im_h, 1:step:im_w, 2);
W = N(1:step:im_h, 1:step:im_w, 3);

shape = figure;
quiver3(X, Y, zeros(size(X)), U, V, W);
%view([0, 90]);
axis off;
axis equal;

drawnow;

end

function N = computeNorms(I, L, M)
p = size(I, 1);
im = I{1};
[im_h, im_w, ~] = size(im);
T = zeros(im_h, im_w, p);

for i = 1:p
    im = I{i};
    for h = 1:im_h
        for w = 1:im_w
            if M(h,w)
                r = im(h,w,1);
                g = im(h,w,2);
                b = im(h,w,3);
                tempInten = norm(double([r g b]));
                T(h,w,i) = tempInten;
            end
        end
    end
end

N = zeros(im_h, im_w, 3);
for h = 1:im_h
    for w = 1:im_w
        if M(h,w)
            inten = reshape(T(h,w,:), [p,1]);
            n = (L.'*L)\(L.'*inten);
            if norm(n) ~= 0
                n = n/norm(n);
            else
                n = [0; 0; 0];
            end
            
            N(h,w,:) = n;
        end
    end
end

end

function drawNormals(x, y, nx, ny, nz, r)
figure, axis on
hold on;

nx = r*nx;
ny = r*ny;

depth = 1.1-nz;

line_x = [x x+nx/depth];
line_y = [y y+ny/depth];
plot(x,y,'g*', 'MarkerSize', 3, 'LineWidth', 1);
line(line_x, line_y);
end

function [xnormal, ynormal, znormal] = normals(p,q)
normy = sqrt(p.^2+q.^2+1);
xnormal = -p./normy;
ynormal = -q./normy;
znormal = 1./normy;
end

function [p,q] = getPQ(img)
[cx, cy] = imgCenter(img);
[py, px] = brightestPoint(img);

x = px - cx;
y = py - cy;
z = sqrt(200^2-x.^2-y.^2);

p = x/-z;
q = y/-z;

end

function [py, px] = brightestPoint(img)
hsvImg = rgb2hsv(img);
v = hsvImg(:,:,3);
high = v==max(v(:));
render = zeros(size(v));
render(high) = 1;
result = bwmorph(render, 'shrink', Inf);
[py, px] = find(result>0);
end

function [cx, cy] = imgCenter(img)
imgSize = size(img);
cy = imgSize(1)/2;
cx = imgSize(2)/2;
end