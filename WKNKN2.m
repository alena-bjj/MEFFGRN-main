function [MD_mat_new] = WKNKN2( MD_mat, MM_mat, DD_mat, r )

[rows,cols]=size(MD_mat);
y_m=zeros(rows,cols);  
y_d=zeros(rows,cols);  

[knn_network_m,KSm] = SNN( MM_mat);  %for miRNA
for i = 1 : rows   
        w=zeros(1,KSm(i));
        [sort_m,idx_m]=sort(knn_network_m(i,:),2,'descend'); 
        sum_m=sum(sort_m(1,1:KSm(i)));   
        for j = 1 : KSm(i)
            w(1,j)=r^(j-1)*sort_m(1,j); 
            y_m(i,:) =  y_m(i,:)+ w(1,j)* MD_mat(idx_m(1,j),:); 
        end                      
            y_m(i,:)=y_m(i,:)/sum_m;              
end


[knn_network_d,KSd] = SNN( DD_mat);  %for disease
for i = 1 : cols   
        w=zeros(1,KSd(i));
        [sort_d,idx_d]=sort(knn_network_d(i,:),2,'descend');
        sum_d=sum(sort_d(1,1:KSd(i)));
        for j = 1 : KSd(i)
            w(1,j)=r^(j-1)*sort_d(1,j);
            y_d(:,i) =  y_d(:,i)+ w(1,j)* MD_mat(:,idx_d(1,j)); 
        end                      
            y_d(:,i)=y_d(:,i)/sum_d;               
end

a1=1;
a2=1;
y_md=(y_m*a1+y_d*a2)/(a1+a2);  

 for i = 1 : rows
     for j = 1 : cols
         MD_mat_new(i,j)=max(MD_mat(i,j),y_md(i,j));
     end    
 end

end

function [ knn_network,ks] = SNN( network )
    [rows, cols] = size( network );
%     network= network-diag(diag(network)); 
    knn_network = zeros(rows, cols);
	% 定义�?个列表或者行向量，用来存储k
	ks =  zeros(rows, 1);
    [sort_network,idx]= sort(network,2,'descend');

    kend = 1/10;
    si=floor(kend*cols);

    for i = 1 : rows 
        kl=0;
        l=floor(sort_network(i,:)*10)/10;
        mk=tabulate(l);
        j=size(mk,1);
        while kl<si
            kf=kl;
            kl=kl+mk(j,2);
            j=j-1;
        end
        
        if kl<2*si
            k=kl;
        elseif kf>si/2
            k=kf;
        else
            k=si;
        end 
		% 把k保存至列表ks
		ks(i,1) = k;
        knn_network(i,idx(i,1:k))=sort_network(i,1:k);
    end
     for i = 1 : rows 
          for j = 1 : cols 
               knn_network(i,j)=max(knn_network(i,j),knn_network(j,i));
          end
     end
end

