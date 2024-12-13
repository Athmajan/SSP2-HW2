%% Lab 2 - Kalman Filter
clear;clc;close all

N = 100;
p = 2;

A1 = [0.99 0;0 0.999];
Q1 = 0.0001*eye(2);


A2 = [0.99 0;0 0.999];
Q2 = 0.01*eye(2);

A3 = [0.99 0.01;0.01 0.999];
Q3 = 0.001*eye(2);

sigma_w= 0.1;

A = A2;
Q = Q2;


%% Fig 13.16 (channel realization)
hn = [1;0.9]; % h[-1]
Hn= [];
Hn= [Hn hn];
for n=1:N+1
    hn = A*hn+(mvnrnd(zeros(p,1),Q))'; %dynamic model
    Hn= [Hn hn];
end
figure;
subplot(2,1,1)
plot(0:N,Hn(1,1:N+1));
ylim([0 2])
ylabel('tap weight,hn[0]')
xlabel('Sample number, n')

subplot(2,1,2)
plot(0:N,Hn(2,1:N+1));
ylim([0 2]);
ylabel('tap weight,hn[1]')
xlabel('Sample number, n')
savefig('Fig_13_16_channel_realization.fig'); % Save the figure as .fig

%% Signal generation (Fig 13.17)
%%%% input v[n] %%%%
T=10;
vn = zeros(N+2,1);
for n=T/2+1:T+1
    vn(n:T:N+1) = 1;
end
figure;
subplot(3,1,1)
plot(0:N,vn(1:N+1)); %Fig 13.17
ylim([-1 3])
ylabel('channel input,v[n]')
xlabel('Sample number, n')
savefig('Fig_13_17_input_vn.fig'); % Save the figure as .fig

%%%% noiseless channel output  y[n] %%%%

hn0 = Hn(1,1:N+1);
hn1 = Hn(2,1:N+1);
Yn= [];

n_values = 2:length(vn);
vn_minus_1 = vn(n_values - 1);
vn_minus_1 = [0 ; vn_minus_1];

for n=1:N+1
    yn = hn0(n)*vn(n) + hn1(n)*vn_minus_1(n);
    Yn= [Yn yn];
end

subplot(3,1,2)
plot(0:N, Yn(1:N+1));% plot y[n] (Fig 13.17b) 
ylim([-1 3]);
ylabel('noiseless channel output, y[n]')
xlabel('Sample number, n')
savefig('Fig_13_17_noiseless_channel_output.fig'); % Save the figure as .fig

%%%% channel output x[n] %%%%
wn = sqrt(sigma_w)*randn(1,N+1);
xn = yn + wn; 

subplot(3,1,3)
plot(0:N,xn) % plot received signal with noise (Fig 13.17b) 
ylim([-1 3]);
ylabel('channel output, x[n]')
xlabel('Sample number, n')
savefig('Fig_13_17_channel_output_xn.fig'); % Save the figure as .fig

%% KALMAN FILTER %%%%%%%%%%%%%%%%%%%

H_hat = zeros(2,N);
M= zeros(p,p,N);
M(:,:,1) = eye(2);
K = zeros(2,N);

for i=2:N
    H_hat(:,i) = A*H_hat(:,i-1);
    M(:,:,i) = A*M(:,:,i-1)*A' + Q;
    vnVec = [vn(i);vn(i-1)];
    K(:,i) = (M(:,:,i)*vnVec)/(sigma_w + vnVec'*M(:,:,i)*vnVec);
    H_hat(:,i) = H_hat(:,i) + K(:,i)*(xn(i) - vnVec'*H_hat(:,i));
    M(:,:,i) = (eye(p) - K(:,i)*vnVec')*M(:,:,i);
end

% Kalman filter estimate
figure;
subplot(2,1,1)
plot(0:N,Hn(1,1:N+1),'k--','LineWidth', 1)
hold on
plot(1:N,H_hat(1,:),'k','LineWidth', 2)
legend('True','Estimate','Location','Best')
ylim([0 2])
ylabel('Tap weight, hn[0]')
xlabel('Sample number, n')
savefig('Kalman_Filter_Tap_Weight_1.fig'); % Save the figure as .fig

subplot(2,1,2)
plot(0:N,Hn(2,1:N+1),'k--','LineWidth', 1)
hold on
plot(1:N,H_hat(2,:),'k','LineWidth', 2)
legend('True','Estimate','Location','Best')
ylim([0 2])
ylabel('Tap weight, hn[1]')
xlabel('Sample number, n')
savefig('Kalman_Filter_Tap_Weight_2.fig'); % Save the figure as .fig

% Kalman Gain
figure;
subplot(2,1,1)
plot(1:N,K(1,1:N),'k','LineWidth', 1)
ylim([-1.5 1.5])
ylabel('Kalman Gain, K1[n]')
xlabel('Sample number, n')
savefig('Kalman_Filter_Kalman_Gain_1.fig'); % Save the figure as .fig

subplot(2,1,2)
plot(1:N,K(2,1:N),'k','LineWidth', 1)
ylim([-1.5 1.5])
ylabel('Kalman Gain, K2[n]')
xlabel('Sample number, n')
savefig('Kalman_Filter_Kalman_Gain_2.fig'); % Save the figure as .fig

% MMSE
figure;
subplot(2,1,1)
plot(1:N,squeeze(M(1,1,:)),'k','LineWidth', 1)
ylim([0 0.2])
ylabel('Minimum MSE, M11[n]')
xlabel('Sample number, n')
custom_ticks = 0:0.02:0.4; % Define your custom tick values
yticks(custom_ticks);
savefig('Kalman_Filter_Minimum_MSE_1.fig'); % Save the figure as .fig

subplot(2,1,2)
plot(1:N,squeeze(M(2,2,:)),'k','LineWidth', 1)
ylim([0 0.2])
ylabel('Minimum MSE, M22[n]')
xlabel('Sample number, n')
custom_ticks = 0:0.02:0.2; % Define your custom tick values
yticks(custom_ticks);
savefig('Kalman_Filter_Minimum_MSE_2.fig'); % Save the figure as .fig
