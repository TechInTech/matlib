%%方法一
x = load('ex2x.dat');
y = load('ex2y.dat');
plot(x,y,'*')
xlabel('height')
ylabel('age')
x = [ones(size(x),1),x];
w=inv(x'*x)*x'*y
hold on
%plot(x,0.0639*x+0.7502) 
plot(x(:,2),0.0639*x(:,2)+0.7502)