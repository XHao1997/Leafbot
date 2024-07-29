%   @autor: fuqb
%   @data:  2022-09-17
%   @function:   PlotRobot.m

function robot=PlotRobot()
%建立机器人模型
% theta d a alpha offset
L1=Link([0 0 0 0 0 ],'modified'); %连杆的D-H参数
L2=Link([0 149.09 0 -pi/2 0 ],'modified');
L3=Link([0 0 431.8 0 0 ],'modified');
L4=Link([0 433.07 20.32 -pi/2 0 ],'modified');
L5=Link([0 0 0 pi/2 0 ],'modified');
L6=Link([0 0 0 -pi/2 0 ],'modified');
robot=SerialLink([L1 L2 L3 L4 L5 L6],'name','puma560','base' , ...
transl(0, 0, 0.62)* trotz(0)); %连接连杆，机器人取名puma560
robot.display();
q=[10*pi/180 0*pi/180 0*pi/180 0*pi/180 0*pi/180 0*pi/180 ];
robot.plot(q);
robot.teach();
% robot.fkine(q)

hold on
% T=TransformMatrix(10*pi/180 ,0, 0 ,0)*TransformMatrix(-10*pi/180 ,149.09, 0, -pi/2)*TransformMatrix(-10*pi/180, 0, 431.8, 0)*...
%     TransformMatrix(10*pi/180, 433.07, 20.32, -pi/2)*TransformMatrix(10*pi/180, 0, 0, pi/2)*TransformMatrix(10*pi/180, 0, 0 ,-pi/2);
% [T,T01,T02,T03,T04,T05,T06]=RobotFkine(q,false);
end
