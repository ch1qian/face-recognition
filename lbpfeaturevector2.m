%��Ӧ5*5ģ������ֵ�γ���������
function [m,f]=lbpfeaturevector2(mat,s,n)
[~,~,d]=size(mat);
if d==3
    mat=rgb2gray(mat);
end
mat=double(mat);
mat=imresize(mat,[100 100],'bicubic');%��һ��

k=calculatelbp2(mat);
f=mat2gray(k);
%imshow(f);
k=uint8(k);
m=[];
for i=1:n:s
    for j=1:n:s
        A=k(i:(i+19),j:(j+19));
        h=[];
        h=imhist(A);
        m=[m;h/(n*n)];
    end
end
