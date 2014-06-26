xps = [0.0, 0.0, 0.1, 0.1, 0.2, 0.0, -0.1, -0.2, -0.2, -0.5];
yps = [0.0, 0.0, 0.1, 0.0, 0.1, -0.1, -0.1, -0.2, -0.5, -0.5];
xs = [1.2, 0.0, 0.0, 1.2, 1.2, 0.0, -1.0, 0.5, 0.1, 0.6];
Rs = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.1];
hs = [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.5, 1.0, 0.5, 1.0];
int_pots = zeros(1, size(xps));

for i=1:size(xps(:))
    int_pots(i) = int_pot(xps(i), yps(i), xs(i), Rs(i), hs(i), 'gauss')
end;

csvwrite('expected_pot_intargs_2D.dat', int_pots)
parameters = [xps; yps; xs; Rs; hs ];
csvwrite('intarg_parameters_2D.dat', parameters  )