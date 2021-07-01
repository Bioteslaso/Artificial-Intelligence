%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
%%%% k = bits of information
%%%% m = bits the redundancy
%%%% n = k+m ; length of the codewords
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% choose the example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
example_num=3;
%%%%% matrix that generates the redundancy
switch example_num
    case 1
            P=[0 1 1; 1 0 1] %%%% dim k x m
    case 2
            P=[0 1 1 1 0; 1 1 1 1 1; 1 0 0 1 0] %%%% dim k x m
    case 3 %%% Hamming code !! (it is a ``perfect'' code: all the column of H are all the possible sindromes)
            P=[1 1 0; 1 0 1; 0 1 1; 1 1 1];    %%%% dim k x m
            %%% all the columns of H=[P' I] corresponds to the numbers 1 to
            %%% 2^m-1 expressed in binary  (in this case 1 to 7)
    case 4  P=[0 1 0 1]; %%%% dim k x m
    case 5  %%% Repetition code !!  super-imp and super-int   
            P=[1 1 1 1]; %%%% dim k x m 
    case 6    
            P=[1 1 1 1; 1 1 1 1]; %%%% dim k x m    
    case 7    
            P=[1 1 1 1; 0 0 0 0]; %%%% dim k x m       
end
[k,m]=size(P);
disp('---------------------------------- ')
disp(' Dimension of P ')
size(P)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% generation matrix %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G=[eye(k,k) P] %%%% dim k x n
[k,n]=size(G);
disp(' Dimension of G ')
size(G)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% parity check matrix %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H=[P' eye(m,m)] %%%% dim m x n
disp(' Dimension of H ')
size(H)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('-------------------------------------')
disp('just a check')
G*H'
mod(G*H',2)
disp('-------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('-------------------------------------')
disp('-------------------------------------')
disp('CODEWORD GENERATION')
disp('----- ----- ----- ----- -----')
disp(' ')
disp('all possible messages !!!')
b = dec2bin(2^k-1:-1:0)-'0' %%% all possible messages to send
 [num_allposs_b,nada]=size(b);
disp(' ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:num_allposs_b
    c(i,:)=mod(b(i,:)*G,2);
end
disp('the corresponding CODEWORDS')
c
disp(' ')

disp('Weigths of all CODEWORDS')
W=sum(c,2)'; %%% sum(c')
W
pos_zero=find(W==0);
WwithoutZero=W;
WwithoutZero(pos_zero)=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('mimimum Hamming distance')
Dmin=min(WwithoutZero)
disp('-------------------------------------')
disp('-------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('syndrome of the messages b - test ')
for i=1:num_allposs_b
    sindromes_of_b(i,:)=mod(c(i,:)*H',2); %%% each sindrome of length m
end
sindromes_of_b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# notice that the error vector must the corresponding dimension of the case chosen at the beginning
 disp(' ')
 c(2,:)
 e=c(2,:) % yields later zero
 e=[1 0 0 0 0 0 0] % change up to one bit to 1 --> syndrome coincides
 r=mod(c(2,:)+e,2)
 disp(' syndrome of r')
     s=mod(r*H',2) %%% each sindrome of length m%
#H
%%%%%%%%%%
disp('--------------------------------------------------------------------------------------------------------------------')
disp('   ')
disp('(the most likely, in terms of prob) SINDROMES corresponding to zero errors or only one error (partial table error-sindrome and number of errors=1)')
disp('   ')
e_now=zeros(1,n);
disp([num2str(e_now),' ',' -------> ', num2str(e_now*H'),' -- num of err --  ', num2str(sum(e_now))])
for i=1:n
    e_now=zeros(1,n);
    e_now(i)=1;
    %%%% solo para control (only for check)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% sum(e_now*H'-H(:,i)') %%% it should be zero always!!!
    %%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%
    disp([num2str(e_now),' ',' -------> ', num2str(H(:,i)'),' -- num of err --  ', num2str(sum(e_now))])
    s_part(i,:)=H(:,i)'; %%% just some of the sindromes correspoding to zero errors or only one error
end
s_part(end+1,:)=zeros(1,m);
%%%%%%%%%%
disp(' ')
disp('Find the remaining possible sindromes... ')
disp(' ')
all_s_now = dec2bin(2^m-1:-1:0)-'0';
[num_all_poss_s,nada]=size(all_s_now);
%%%%
for i=1:num_all_poss_s
    for j=1:n+1      
         aux(j)=sum(abs(all_s_now(i,:)-s_part(j,:))); 
    end
    aux2=min(aux);
     if aux2~=0
             disp(['  ',num2str(all_s_now(i,:))])
         end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('--------------------------------------------------------------------------------------------------------------------')
%disp('-------------------------------------------')
all_e = dec2bin(2^n-1:-1:0)-'0';
[num_all_poss_e,nada]=size(all_e);
disp('   ')
disp('COMPLETE TABlE ERROR - SINDROME (and number of errors)')
disp('   ')
for i=1:num_all_poss_e
    all_s(i,:)=mod(all_e(i,:)*H',2);
    %%%%%%%%%%%%
    disp([num2str(all_e(i,:)),' ',' -------> ', num2str(all_s(i,:)),' -- num of err --  ', num2str(sum(all_e(i,:)))])
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('   ')
disp('--------------------------------------------------------------------------')
disp([' RATE of the code: ',num2str(k/n),' where k= ',num2str(k),' and n= ',  num2str(n)])
disp('--------------------------------------------------------------------------')
disp(' RECEIVED SEQUENCE OF BITS: ')
switch example_num
    case 1
         r=[1 0 1 0 1]
         %r=c(2,:)
         sindr_example_after_received=mod(r*H',2)
    case 2
         %r=[1 0 1 0 1 0 0]
         r=c(3,:)         
         r=[ 1   0   1   1   1   1   0   1]
         sindr_example_after_received=mod(r*H',2)
    case 3
         r=[1 0 1 1 1 1 1]
          %r=c(1,:)
        sindr_example_after_received=mod(r*H',2) 
    case 4
           r=[0 1 1 1 0]
          %r=c(1,:)
           sindr_example_after_received=mod(r*H',2) 
    case 5
           r=[0 1 1 1 0]
          %r=c(1,:)
           sindr_example_after_received=mod(r*H',2)
    case 6
           %r=[0  0  1  1  1  0]
            r=[1  0  0  0  0  1]
          %r=c(1,:)
           sindr_example_after_received=mod(r*H',2)    
    case 7
           r=[0  0  1  1  1  0]
           % r=[1  0  0  0  0  1]
          %r=c(1,:)
           sindr_example_after_received=mod(r*H',2)    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp(' what happens ?  (Table "error - sindrome- num. errors" )')
disp(' ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
poss_e_pattern=[];
num_err_detection=[];
%%%%%%%%%%%
for i=1:num_all_poss_e
  %%%%  
    aux=sum(abs(all_s(i,:)-sindr_example_after_received));
  %%%%%
    if aux==0  
        poss_e_pattern(end+1,:)= all_e(i,:);
        disp([num2str(all_e(i,:)),' ',' -------> ', num2str(all_s(i,:)),' -- num of err --  ', num2str(sum(all_e(i,:)))])
        num_err_detection(end+1)=sum(all_e(i,:));
    end
    %%%%%%%%%%%%
end
disp(' ')
disp(['number of error patterns associated to the same sindrome = ',num2str(2^n/2^m)])
disp('----------------------------------------------------------------------------')
disp(' ')
disp([' POSSIBLE SENT CODEWORDS (estimated codewords) IN ORDER OF LIKELIHOOD '])
p_channel=0.05;
disp(['   in a binary symmetric channel: (probability computed with p = ',num2str(p_channel),' )'])
disp(' ')
[aux,aux_pos]=sort(num_err_detection);

for i=1:length(aux_pos)
     c_est(i,:)=mod(r-poss_e_pattern(aux_pos(i),:),2);
     ps(i)=p_channel^(num_err_detection(aux_pos(i)))*(1-p_channel)^(n-num_err_detection(aux_pos(i)));
     disp([' ',num2str(c_est(i,:)),'--->',' with prob. = ',num2str(ps(i))])        
end
disp(' ')
disp([' ...note that pobabilties do not sum 1....since we are not considering '])
disp([' ...all the vectors of error patterns....BUT JUST THE POSSIBLE ONES !!! '])
disp(' ')
disp([' then what is the most probable message sent? '])
disp(' ')

disp('----------------------------------------------------------------------------')