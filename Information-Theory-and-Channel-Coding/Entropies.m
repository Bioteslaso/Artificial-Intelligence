clear all
close all
clc

# define Join probability 
# enter the matrix JP you want to study!!
JP = [1/8,1/16,1/32,1/32;1/16,1/8,1/32,1/32;1/16,1/16,1/16,1/16;1/4,0,0,0];

##example ##JP = [1/8,1/16,1/32,1/32;1/16,1/8,1/32,1/32;1/16,1/16,1/16,1/16;1/4,0,0,0];

###################  CODE  #####################
size(JP);
'correct?'
sum(sum(JP))
disp('------------ ENTROPIES --------------')

%marginal pmfs. Sum adds the values by columns and gives a matrix 1xD with
%the value of each sum.
marginalp_x = sum(JP);
marginalp_y = sum(JP'); %also could be done by (JP,2)

for i=1:4 % fixed
  for j=1:4 # iterated
    Condp_XY(j,i) = JP(j,i)/marginalp_y(i);
    Condp_YX(i,j) = JP(i,j)/marginalp_x(j);
   end
end
##for j=1:4
##    for i=1:4
##        Condp_XY(i,j) = JP(i,j)/marginalp_x(j);
##        Condp_YX(j,i) = JP(j,i)/marginalp_y(j);
##    end    
##end

% All possible entropies

##Entropy of X and Y
for k=1:4
    Entropy(k) = -marginalp_x(k).*log2(marginalp_x(k));
    Entropy2(k) = -marginalp_y(k).*log2(marginalp_y(k));
end


HX = sum(Entropy)
HY = sum(Entropy2) 


% joint entropy
for j = 1:4
    for i=1:4
        if JP(i,j) == 0; % avoid Nan errors
          JP(i,j) = 1;
        end
        jointent(j,i) = -JP(j,i).*log2(JP(j,i));
        
    end
end
HJoint = sum(sum(jointent))

% conditional entropies
for j=1:4
    for i=1:4
        a = Condp_YX(j,i);
        b = Condp_XY(j,i);
        if Condp_YX(j,i)== 0 % correct NaN inside log2
          a = 1;
        end
        if Condp_XY(j,i)== 0 % correct NaN inside log2
          b = 1;
        end
        HCondYX(j,i) = -JP(j,i).*log2(a);
        HCondXY(j,i) = -JP(j,i).*log2(b);
    end
end

HCondYX = sum(sum(HCondYX))
HCondXY = sum(sum(HCondXY))
mutual_info = HX - HCondXY
'>> check in the figure how every inequelity of the slides is fulfilled'

# Let's see if everything makes sense:
clf;
y = [HCondYX HCondXY mutual_info; HJoint 0 0];
h = bar (y, "stacked");
