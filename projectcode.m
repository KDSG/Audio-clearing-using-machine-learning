

lambda = 1 ;

[x1,fx1] = audioread('m1.wav');
x1 = x1(1:53556,1);
[x2,fx2] = audioread('timit_denoising_groundtruth_noise.wav');
[y,fy] = audioread('o1.wav');
y = y(1:53556,1);

hold on;
y = y';
X = [x1 ; x2];
X = X';

%[x1,fx1] = audioread('.wav');
%x1 = x1(1:53556,1);
%[y,fy] = audioread('.wav');
%y = y(1:53556,1);



sound(x1,fx1);
plot(x1);
%pause;
sound(x2,fx2);
plot(x2)
%pause;
%sound(y,fy);
%plot(y);
input_layer_size  = length(X);  
hidden_layer_size = 25;   
num_labels = length(x1); 

Theta1 = [ones(25,length(X)+1)];
Theta2 = [ones(length(x1),26)];

Theta1 = rand(hidden_layer_size,1+input_layer_size);
Theta2 = rand(num_labels,1+hidden_layer_size);

nn_params = [Theta1(:) ; Theta2(:)];

options = optimset('MaxIter', 30);

Theta1 = rand( hidden_layer_size,1+input_layer_size);
Theta2 = rand(num_labels,1+hidden_layer_size);

initial_nn_params = [Theta1(:) ;Theta2(:)];
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
size(nn_params)
hidden_layer_size
input_layer_size
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));      
     
fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');

pause;

%[x1,fx1] = audioread('timit_denoising_mixture.wav');
%x1 = x1(1:53556,1);
%sound(x1,fx1);
%X = [x1 ; x2];
%X = X';
m= size(X,1);
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

sound(h2,fx1);
%audiowrite('abc1.wav',h2,fx1);
plot(h2);

     
                 
                 