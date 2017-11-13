% PROGRAM NAME: HW4_Aiyagari_PFI
clear, clc
tic

% parameters
beta = .99; %discount factor 
sigma = 2; % coefficient of risk aversion
alpha=1/3; % Capital Share
delta= .025; % depreciation
rho =.5; % AR(1) persistence
sigma_e=.2; % White noise variance

num_z = 5;
[z_grid, PI]= TAUCHEN(num_z, rho, sigma_e, 3);
z_grid =exp(z_grid');

% Invariant Transition matrix
[eig_vec, eig_val]= eig(PI');

% Normalize first eigen-vector to have a sum of 1.
PI_norm= eig_vec(:,1) ./ sum(eig_vec(:,1));

% ASSET VECTOR
a_lo = 0; %lower bound of grid points
a_hi = 80; %upper bound of grid points
num_a = 50;

a = linspace(a_lo, a_hi, num_a); % asset (row) vector

% Aggregate Labor and Capital

% Agg Labor is integral of z*L*PI_norm over z, L=1
totL= z_grid*PI_norm;

% Capital Range 
k_min = 20;
k_max = 50;

K_dist=1;

while abs(K_dist) >= .5
    
    % Set INITIAL K_GUESS, rental rate and wage
    k_guess = (k_min + k_max)/2;
    
    r= alpha*((totL/k_guess)^(1-alpha)) + (1-delta);
    w= (1-alpha)*((k_guess/totL)^alpha);
    
    cons = bsxfun(@minus, r*a', a);
    cons = bsxfun(@plus, cons, permute(z_grid, [1 3 2])*w);
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret(cons<0) = -Inf;
    
    
    v_guess = zeros(num_z, num_a);
    
    % Policy Function Iteration
    
    k=30; 
    v_dist=1;
    
    while abs(v_dist) > 0.000001
   % CONSTRUCT TOTAL RETURN FUNCTION
   v_mat = ret + beta * ...
       repmat(permute(PI * v_guess, [3 2 1]), [num_a 1 1]);
   
   % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
   [vfn, pol_indx] = max(v_mat, [], 2);
   vfn = permute(vfn, [3 1 2]);
   
   pol_indx = permute(pol_indx, [3 1 2]); 
   
   v_dist = abs(max(v_guess(:) - vfn(:)));
   
   v_guess = vfn; %update value functions

   %Q Matrix
   Q= makeQmatrix(pol_indx,PI);
   
   pol_fn = a(pol_indx);
   
   u_mat= bsxfun(@minus, r*a, pol_fn);
   u_mat= bsxfun(@plus, u_mat, z_grid'*w);
   u_mat= (u_mat .^(1-sigma)) ./(1-sigma);
   u_vec= u_mat(:);
   
   w_vec= v_guess(:);
   
   for j = 1:k
       
       w_vecN= u_vec + beta*Q*w_vec;
       w_vec= w_vecN;
   end
   
   v_guess = reshape(w_vec, num_z, num_a);
    end
    
  % Distribution
Mu = zeros(num_z,num_a);
Mu(1, 4) = 1; % initial guess: everyone employed, 0 assets

% ITERATE OVER DISTRIBUTIONS
mu_tol = 1;
while mu_tol > 0.0000001
    [emp_ind, a_ind] = find(Mu > 0); % find non-zero indices
    
    MuNew = zeros(size(Mu));
    for ii = 1:length(emp_ind)
        apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); 
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
            (PI(emp_ind(ii), :) * Mu(emp_ind(ii), a_ind(ii)) )';
    end

    mu_tol = max(abs(MuNew(:) - Mu(:)));
    
    Mu = MuNew ;
end  
    
    
%Aggregate Savings

aggsav = sum( pol_fn(:) .* Mu(:) ); % Aggregate future assets   

K_dist = aggsav - k_guess;

if K_dist > 0 ;
    k_min = k_guess ;
end ;
if K_dist < 0;
    k_max = k_guess ;
end ;
    
display (['Current K Guess = ', num2str(k_guess)])
display (['Aggregate desired wealth = ', num2str(aggsav)]);
display (['New Kmin is ', num2str(k_min), ', New Kmax is ', num2str(k_max)]);
display (['New K Guess is ', num2str((k_max + k_min)/2)]);    
display (['Tolerance Level ', num2str(K_dist)]);     

end


figure(1)
 plot(a,vfn(1,:))
 hold on
 plot(a,vfn(2,:))
 hold on
 plot(a,vfn(3,:)')
 hold on
 plot(a,vfn(4,:))
 hold on
 plot(a,vfn(5,:))
 legend('Z= 0.5','Z= 0.7','Z= 1','Z= 1.4', 'Z= 1.9','northwest')
 title(['Value Function'])
 hold off

figure(2)
plot(a,pol_fn(1,:))
hold on
plot(a,pol_fn(2,:))
hold on
plot(a,pol_fn(3,:))
hold on
plot(a,pol_fn(4,:))
hold on
plot(a,pol_fn(5,:))
hold on
legend=('z=0.5''z=0.5''z=0.5''z=0.5''z=0.5''northwest')
hold on
title('Policy Function')
hold off

r_eqlm=1/beta %Competitive Rate. r is model rate

pop=reshape(Mu',[num_z*num_a,1])
w=reshape(repmat(a,num_z,1)',[num_z*num_a,1])

figure(3)
gini_w=gini(pop,w,true)
title(['Wealth Inequality, Gini=',num2str(gini_w)])

mu=sum(Mu)
figure(4)
bar(a,mu)
title('Wealth Distribution')

runtime=toc