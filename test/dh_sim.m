%           theta    d      a    alpha    offset
L1=Link([    0       64      0      0         0    ], 'modified');
L2=Link([    -pi/2       0      0      -pi/2     -pi/2], 'modified');
L3=Link([    0       0      43.5      0         0    ], 'modified');
L4=Link([    0       0      82.85     0         0    ], 'modified');
L5=Link([    0       82.85      0      -pi/2     -pi/2], 'modified');
L6=Link([    0       0      0      -pi/2     0    ], 'modified');

robot=SerialLink([L1,L2,L3,L4,L5,L6]); % 将四个连杆组成机械臂

% p= robot.fkine(q);

% robot.name='dofbot';
% robot.display();
% robot.teach();
% robot.plot([0 L2.theta 0 0 L5.theta 0])
p = robot.fkine([0 -pi/2 0 -pi/2 0 0])

% 
% N=30000;    %随机次数
% 
%     %关节角度限制
% limitmax_1 = pi/2;
% limitmin_1 = -pi/2;
% limitmax_2 = pi/2;
% limitmin_2 = -pi/2;
% limitmax_3 = pi/2;
% limitmin_3 = -pi/2;
% limitmax_4 = pi/2;
% limitmin_4 = -pi/2;
% 
% theta1=L1.theta+(limitmin_1+(limitmax_1-limitmin_1)*rand(N,1)); %关节1限制
% theta2=L2.theta+(limitmin_2+(limitmax_2-limitmin_2)*rand(N,1)); %关节2限制
% theta3=L3.theta+(limitmin_3+(limitmax_3-limitmin_3)*rand(N,1)); %关节3限制
% theta4=L4.theta+(limitmin_4+(limitmax_4-limitmin_4)*rand(N,1)); %关节4限制
% 
% for n=1:1:3000
% qq=[theta1(n),theta2(n),theta3(n),theta4(n), 0, 0];
% robot.plot(qq);%动画显示
% Mricx=robot.fkine(qq);
% 
% zlim tight
% plot3(Mricx.t(1),Mricx.t(2),Mricx.t(3), 'r.','MarkerSize',4);%画出落点
% hold on;
% end
