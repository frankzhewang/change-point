function [ cost, time ] = scarf(id, k, a, S, T, h, p, step)
% SCARF computes the expected total cost of a gamma-gamma Bayesian 
%   inventory problem in Scarf (1960).
%
% Input:
%   id: string. instance identifier.
%   k: scalar. demand shape parameter
%   a: scalar. prior shape parameter
%   S: scalar. prior scale parameter
%   T: scalar. length of the planning horizon
%   h: scalar. unit holding cost
%   p: scalar. unit penalty cost
%   step: scalar. step size for discretizing the state space
%
% Output:
%   cost: expected total cost
%   time: computation time
%   scarf-<id>.mat

% h = 1;
% step = .01;

tic;

% TODO: only need to compute the standardized Scarf problem once for all
% the instances with identical k, a, T, h, p
[~,j,~,~] = stdScarfGmGm(k, a, T, h, p, step);

cost = j*S;
time = toc;

save(['scarf-' id '.mat'],...
     'k','p','a','S','T','h','p','step','cost','time');
