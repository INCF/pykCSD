elec_pos = [0, 0 ; 0, 1 ; 1, 0 ; 1, 1; 0.5, 0.5]
pots = [0,; 0,;0,; 0,; 1,;]
k = kcsd2d(elec_pos, pots, 'manage_data', 0, 'n_src', 9, 'gdX', 0.1, 'gdY', 0.1)
k.estimate_potentials
k.estimate %csd

csvwrite('five_elec_elecs.dat', k.el_pos)
csvwrite('five_elec_pots.dat', k.pots)

csvwrite('five_elec_dist_max.dat', k.dist_max)
csvwrite('five_elec_R.dat', k.R)
csvwrite('five_elec_kpot.dat', k.K_pot)
csvwrite('five_elec_dist_table.dat', k.dist_table)
csvwrite('five_elec_interp_pot.dat', k.interp_pot)
csvwrite('five_elec_b_src_matrix.dat', k.b_src_matrix)
csvwrite('five_elec_estimated_pot.dat', k.pots_est)
csvwrite('five_elec_estimated_csd.dat', k.CSD_est)
