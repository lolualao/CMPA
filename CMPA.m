close all
clear all
clc

I_s=0.01e-12;
I_b=0.1e-12;
V_b=1.3;
G_p=0.1;


current= @(V) I_s.*(exp(1.2/0.025.*V)-1)+ G_p.*V-I_b.*(exp(-1.2/0.025.*(V+V_b)-1));

V=linspace(-1.95,0.7,200);
I_1=current(V);
I_N=I_1+I_1.*(0.4*rand(1,length(I_1))-0.2);


% figure(1)
% plot(V,I_1)
% 
% figure(2)
% semilogy(V,I_1)


figure(1)
plot(V,I_N)

figure(2)
semilogy(V,I_N)


%% Polynomial fitting
P_4=polyfit(V,I_N,4);
P_8=polyfit(V,I_N,8);



%% Plot graphs
figure(3)
hold on;
plot(V,I_N)
plot(V,polyval(P_4,V));
plot(V,polyval(P_8,V));
hold off;
legend('Raw','4O poly','8O poly');

figure(4)
semilogy (V,abs(I_N));
hold on;
semilogy(V,abs(polyval(P_4,V)));
semilogy(V,abs(polyval(P_8,V)));
hold off;
legend('Raw','4O poly','8O poly');

%% Fit with Function
x=V;
% A=0.01e-12;
% C=0.1e-12;
% D=1.3;
% B=0.1;
%f1 = @(x) A.*(exp(1.2/0.025.*x)-1)+B.*x-C.*(exp(-1.2/0.025.*(x+D)-1));

%a
fo_a=fittype('A.*(exp(1.2/0.025.*x)-1)+0.1.*x-C.*(exp(-1.2/0.025.*(x+1.3)-1))');
ff_a = fit(V',I_N',fo_a);
If_a = ff_a(x);

%b
fo_b=fittype('A.*(exp(1.2/0.025.*x)-1)+B.*x-C.*(exp(-1.2/0.025.*(x+1.3)-1))');
ff_b = fit(V',I_N',fo_b);
If_b = ff_b(x);

%c
fo_c=fittype('A.*(exp(1.2/0.025.*x)-1)+B.*x-C.*(exp(-1.2/0.025.*(x+D)-1))');
ff_c = fit(V',I_N',fo_c);
If_c = ff_c(x);

figure(5)
hold on;
plot(V,I_N);
plot(x,If_a);
plot(x,If_b);
plot(x,If_c);
hold off;
legend('Raw','AC','ABC','ABCD');


figure(6)
semilogy(V,abs(I_N));
hold on;
semilogy(x,abs(If_a));
semilogy(x,abs(If_b));
semilogy(x,abs(If_c));
hold off;
legend('Raw','AC','ABC','ABCD');

current= @(V) I_s.*(exp(1.2/0.025.*V)-1)+ G_p.*V-I_b.*(exp(-1.2/0.025.*(V+V_b)-1));

V=linspace(-1.95,0.7,200);
I_1=current(V);
I_N=I_1+I_1.*(0.4*rand(1,length(I_1))-0.2);

inputs = V;
targets = I_N;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs

%% Plot graphs
figure(7)
hold on;
plot(V,I_N);
plot(V,Inn);
hold off;
legend('Raw','nn');


figure(8)
semilogy(V,abs(I_N));
hold on;
semilogy(V,abs(Inn));
hold off;
legend('Raw','nn');




