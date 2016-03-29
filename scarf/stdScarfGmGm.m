function [ y, j, Y, J ] = stdScarfGmGm(k, a, T, h, p, step)
% STDSCARFGMGM solves the standardized gamma-gamma demand Bayesian 
%   inventory problem in Scarf(1960).
%
% Notation:
%   Nx: # of possible inventory levels
%
% Input:
%   a: scalar. shape parameter of the prior distribution
%   k: scalar. shape parameter of the demand distribution
%   T: scalar. # of periods in the planning horizon
%   h: scalar. unit holding cost per period
%   p: scalar. unit penalty cost per period
%   step: scalar. step size for discretizing the state space
%
% Output:
%   y: scalar. optimal base-stock level in the 1st period
%   j: scalar. optimal expected total cost
%   Y: Nx by T matrix. Y(x,t) is the optimal base-stock level
%       in period t with inventory state index x
%   J: Nx by T+1 matrix. J(x,t) is the optimal expected
%       cost-to-go for periods t,...,T given demand inventory state index x

% critical ratio
CR = p/(h+p);

% search for an upper bound on the inventory level
yMax = step;
while betainc(yMax/(1+yMax),k,a) < CR
    yMax = yMax * 2;
end

% possible inventory levels
X = [floor(-1/step) : ceil(yMax/step)] .* step;
[~,Nx] = size(X);

J = zeros(Nx, T+1);
Y = zeros(Nx, T);

% terminal costs
J(:,T+1) = 0; 

% index of inventory level 0
xi0 = 0 - floor(-1/step) + 1;

%% computation using the transformed Bellman equations in Chen (2010)

% backward induction
for t = T:-1:1
    pdf = @(u) (u.^(k-1)) .* (1-u).^((t-1)*k+a-2) ./ beta(k,(t-1)*k+a-1);

    % starting from the inventory level upper bound
    yopt = X(Nx);
    cost = p*(k/((t-1)*k+a-1)-yopt) ...
        + (h+p)*yopt*betainc(yopt/(yopt+1),k,(t-1)*k+a) ...
        - (h+p)*k/((t-1)*k+a-1)*betainc(yopt/(yopt+1),k+1,(t-1)*k+a-1);
    integrand = @(u) ...
        J(round((yopt.*(1-u)-u)./step) - floor(-1/step) + 1,t+1)' ...
            .* pdf(u);
    jmin = cost + ((t*k+a-1)/((t-1)*k+a-1))*integral(integrand,0,1);
    Y(Nx,t) = yopt;
    J(Nx,t) = jmin;
    
    % search for optimal base-stock level yopt
    iopt = Nx;
    for i = Nx-1 : -1 : xi0
        ycand = X(i);
        cost = p*(k/((t-1)*k+a-1)-ycand) ...
            + (h+p)*ycand*betainc(ycand/(ycand+1),k,(t-1)*k+a) ...
            - (h+p)*k/((t-1)*k+a-1)*betainc(ycand/(ycand+1),k+1,(t-1)*k+a-1);
        integrand = @(u) ...
            J(round((ycand.*(1-u)-u)./step) - floor(-1/step) + 1,t+1)' ...
            .* pdf(u);
        jcand = cost + ((t*k+a-1)/((t-1)*k+a-1))*integral(integrand,0,1);
        if jcand > jmin
            break
        else
            Y(i,t) = ycand;
            J(i,t) = jcand;
            jmin = jcand;
            yopt = ycand;
            iopt = i;
        end
    end
        
    % it is optimal to order up to yopt for inventory levels <= yopt
    Y(1:iopt,t) = yopt;
    J(1:iopt,t) = jmin;
end

y = Y(xi0,1);
j = J(xi0,1);
end

