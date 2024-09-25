syms t g a l3 l2 l1 x y;

G_e3 = [0 1 l3;-1 0 0;0 0 1];
G_32 = [cos(t) -sin(t) l2;sin(t) cos(t) 0;0 0 1];
G_21 = [cos(g) -sin(g) l1;sin(g) cos(g) 0;0 0 1];
G_1s = [cos(a) -sin(a) x;sin(a) cos(a) y;0 0 1];

G_es = G_1s * G_21 * G_32 * G_e3;

disp('The symbolic result of G_es is:');
disp(G_es);

values = [t, g, a, l3, l2, l1, x, y];
numerics = [-pi/3, pi/4, pi/4, 1, 1, 1, 1, 1];
G_es_num = subs(G_es, values, numerics);

disp('The numeric result of G_es is:');
disp(G_es_num);

disp('The numeric result of G_es for p1 is:');
disp(G_es_num * [0;0;1]);

disp('The numeric result of G_es for p2 is:');
disp(G_es_num * [1;2;1]);


