function hw6new()

close all;
format compact;

lightDir = load('light_directions.txt');
img1 = imread('4-1.tiff');
img2 = imread('4-2.tiff');
img3 = imread('4-3.tiff');
img4 = imread('4-4.tiff');
img5 = imread('4-5.tiff');
img6 = imread('4-6.tiff');
img7 = imread('4-7.tiff');
%imgArr = {img1; img2; img3; img4; img5; img6; img7};


light1=lightDir(1,:);
light2=lightDir(2,:);
light3=lightDir(3,:);
light4=lightDir(4,:);
light5=lightDir(5,:);
light6=lightDir(6,:);
light7=lightDir(7,:);

% light array
lights= [light1; light2; light3; light4; light5; light6; light7];

% normals
b=ones(400, 400,3);
b=double(b);

% z to find p and q
p=ones(400, 400);
p=double(p);
q=p;
% depth map
depth=ones(400, 400);
depth=double(depth);


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)


for i=1:400
    for j=1:400
        %normal vector calculation
        
        %brightness
        E=[img1(i,j) img2(i,j) img3(i,j) img4(i,j) img5(i,j) img6(i,j), img7(i,j)];
        E=double(E');
        
        tb= (inv(lights'*lights))*lights'*E;
        
        nbm = norm(tb);
        if( nbm == 0)
            b(i,j,:) = 0; 
        else
            b(i,j,:) = tb / nbm; % normal
        end
        
        % to compute z
        tM = [b(i,j,1) b(i,j,2) b(i,j,3)];
        nbm = norm(tM);
        if( nbm == 0)
            tM = [0 0 0];
        else
            tM = tM / nbm;  
        end        
        p(i,j)=tM(1,1);
        q(i,j)=tM(1,2);
    end
end


% create Z using pq
for i=1:400
    for j=1:400
        depth(i,j) = sum(q(1:i, 1)) + sum(p(i,1:j));
    end
end
% depth reverse
%depth = depth*-1;

figure(1);
hold on;
for i=1:15:400
    for j=1:15:400
        % plot normals
        plot3(j+b(i,j,1),i+b(i,j,2),b(i,j,3),'b*');
    end
end
hold off;

figure(2);
mesh(depth);
shading interp
colormap bone


% reverse program

nuNorms = b;
nuZ = depth*-1;
nup=ones(400, 400);
nup=double(p);
q = p;


for i=2:399
    for j=2:399
        
        t = [i;j-1;nuZ(j-1,i)];
        l = [i-1;j;nuZ(j,i-1)];
        c = [i;j;nuZ(j,i)];
        d = cross(l-c, t-c);
        n = norm(d);
        nuNorms(j,i) = n;
        
    end
end
figure(3);
hold on;
for i=1:15:400
    for j=1:15:400
        % plot normals
        plot3(j+nuNorms(i,j,1),i+nuNorms(i,j,2),nuNorms(i,j,3),'b*');
    end
end
hold off;

end

