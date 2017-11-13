%PS4 Aiyagari VFI
clear, clc
tic

%Parameters
beta=.99   %discount factor
sigma=2    %coefficient of risk aversion
alpha=1/3  %Capital Share
delta=0.025 %depreciation rate
rho_e=0.5  %TFP shock persistence
sigma_e=0.2 %White Noise variance
num_z=5

[z_grid,PI]=TAUCHEN(num_z, rho_e, sigma_e,3)

[eig_vec,eig_al]=eig(PI')

%normalize
PI_norm=eig_vec(:,1)./sum(eig_vec(:,1))

%Asset Vector
a_lo=0
a_hi=100
num_a=100
a=linspace(a_lo, a_hi, num_a) %assets

%Labor and capital
totL=zgrid*PI_norm %Integral of zLPi_norm over z, L=1
totL=z_grid*PI_norm

k_min=20
k_max=50
K_dis=1

while abs(K_dis)>=0.01
    
   k_guess = (k_min+k_max)/2;
   r = alpha*((totL/k_guess)^(1-alpha))+(1-delta);
   w = (1-alpha)*(k_guess/totL)^alpha
   
   cons = bsxfun(@minus, r*a',a);
   cons = bsxfun(@plus, cons, permute(z_grid, [1,3,2])*w);
   ret = cons.^(1-sigma)./(1-sigma);
   ret(cons(0))=-Inf;
   
   v_guess = zeros(num_z,num_a);
   
   tol_v=1;
end 
   %%Value Function
   
   while tol_v>0.0001;
       
      value_mat=ret+beta*repmat(permute(PI*v_guess,[3 2 1]),[num_a 1 1]);
      
      %Maximize
      [vfn,pol_indx]=max(v_mat,[],2);
      vfn=permute(vfn,[3 1 2]);
      
      tol_v=abs(max(v_guess(:)-vfn(:)));
      v_guess=vfn;
     
   end
   
   pol_indx = permute(pol_indx,[3 1 2]);
   pol_fn = a(pol_indx);
   
   %Set up Initial Dist
   Mu=zeros(num_z,num_a);
   Mu(1,4)=1
   
   mu_tol=1
   while mu_tol>0.000001
       [state_ind,a_ind]=find(Mu>0);
       newMu=zeros(size(Mu));
       
       for j = 1:length(state_ind)
       apr_ind = pol_indx(emp_ind(j),a_ind(j))
       newMu(:,apr_ind)=newMu(:,apr_ind)+ ...
           (PI(state_ind(j),:))*(Mu(state_ind(j),a_ind(j)))';
       end
       mu_tol=max(abs(newMu(:)-Mu(:)));
       Mu=newMu
          
   end   
   
  AggS = sum( pol_fn(:) .* Mu(:) ); % Aggregate future assets   
