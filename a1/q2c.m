syms time t(time) g(time) a(time) l3 l2 l1 x y pos_x pos_y;

% Transformation matrices
G_e3 = [0 1 l3; -1 0 0; 0 0 1];
G_32 = [cos(t), -sin(t), l2; sin(t), cos(t), 0; 0 0 1];
G_21 = [cos(g), -sin(g), l1; sin(g), cos(g), 0; 0 0 1];
G_1s = [cos(a), -sin(a), x; sin(a), cos(a), y; 0 0 1];

% Compute the combined transformation matrix G_2s
G_2s = G_1s * G_21;

disp('Symbolic G_2s:');
disp(G_2s);

% Substitute numeric values
values = [l1, l2, l3, x, y];
numerics = [1, 1, 1, 0, 0];
G_2s_num = subs(G_2s, values, numerics);

disp('Numeric G_2s:');
disp(G_2s_num);

G_2s_dot = diff(G_2s_num, time);

disp('G_2s_dot matrix:');
disp(G_2s_dot);

p_vec = [pos_x; pos_y; 1];

disp('Velocity result:');
disp(G_2s_dot * p_vec);
