srcs = [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, -0.1, -0.1, -0.1, -0.1, 2.0, -3.0, 4.0, 5.0]
args = [0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.2, 0.2, 0.5, 0.5, 0.7, 0.7, 1.0, 0.5, 0.2, -0.2, 5.0, 0.5, 10., 10.]
curr_pos = [0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, .4, .4, .4, 4.0, 1.0, 1.0, 10.0, 10.0]
hs = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
Rs = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.6, 0.5, 0.1, 0.2, 0.3, 0.1, 1.0, 2.0, 3.0, 4.0]
sigmas = [0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.2, 2.0, 0.2, 2.0, 0.2, 2.0, 0.2, 2.0, 0.2, 2.0]

pot_intargs = zeros(1,20)
for i = i:20
    pot_intargs(i) = pot_intarg(srcs(i), args(i), curr_pos(i), hs(i), Rs(i), sigmas(i), 'gauss')
end

csvwrite('expected_pot_intargs.dat', pot_intargs)
parameters = [srcs; args; curr_pos; hs; Rs; sigmas ]
csvwrite('intarg_parameters.dat', parameters  )