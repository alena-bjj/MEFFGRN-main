function [DD] = Reward1(Y)
[rows,cols]=size(Y);

% REWARD MATRIX
e=9;
DD=zeros(rows,rows);
 for i = 1 : rows
     for j = 1 : rows
         k=0;
        % k = k+1;         
         for s=1:cols
              if(i==s&&Y(j,s)==1)||(Y(i,s)==1&&j==s)
                 k = k+1;
                 DD(i,j)=DD(i,j)+e;   
                 DD(j,i)=DD(j,i)+e;
             elseif (Y(i,s)==0&&Y(j,s)==0&&i~=s&&j~=s)||(Y(i,s)==1&&Y(j,s)==1&&i~=s&&j~=s)
                 k = k+1;
                 DD(i,j)=DD(i,j)+1;
                DD(j,i)=DD(j,i)+1;
                 
              elseif (Y(i,s)~=Y(j,s)&&i~=s&&j~=s)
                  k = k+1;
                 DD(i,j)=DD(i,j)-1;
               DD(j,i)=DD(j,i)-1;
             end             
         end
     end    
 end
 
 MM=zeros(cols,cols);

 for i = 1 : cols
     for j = 1 : cols
         k=0;
         for t=1:rows
             if (i==t&&Y(t,j)==1)||(Y(t,i)==1&&j==t)
                 k=k+1;
                 MM(i,j)=MM(i,j)+e;
                MM(j,i)=MM(j,i)+e;
             elseif (Y(t,i)==1&&Y(t,j)==1&&i~=t&&j~=t)||(Y(t,i)==0&&Y(t,j)==0&&i~=t&&j~=t)
                 k = k+1;
                 MM(i,j)=MM(i,j)+1;
                 MM(j,i)=MM(j,i)+1;
                 
             elseif (Y(t,i)~=Y(t,j)&&i~=t&&j~=t)
                 k = k+1;
                 MM(i,j)=MM(i,j)-1;
                 MM(j,i)=MM(j,i)-1;
             end
         end
     end    
 end


M=MM;
D=DD;

    for i = 1 : size(MM,1)  
        M(i,i)=0;
    end
    for i = 1 : size(DD,1)  
        D(i,i)=0;
    end
 
    for i = 1 : size(MM,1)  
            for j = 1 : size(MM,2)  
                
               norm=(max(M(i,:))-min(M(i,:)))*(max(M(j,:))-min(M(j,:)));
               MM(i,j)=(M(i,j)-min(M(i,:)))*(M(i,j)-min(M(j,:)))/norm;

               MM(i,j)=MM(i,j).^2;
            end
    end
     
    for i = 1 : size(DD,1)
            for j = 1 :  size(DD,2)
               
               norm=(max(D(i,:))-min(D(i,:)))*(max(D(j,:))-min(D(j,:)));
               DD(i,j)=(D(i,j)-min(D(i,:)))*(D(i,j)-min(D(j,:)))/norm;
              DD(i,j)=DD(i,j).^2;
            end
   end
                       
     for i = 1 : size(MM,1)  
        MM(i,i)=1;
    end
    for i = 1 : size(DD,1)  
        DD(i,i)=1;
    end

end
