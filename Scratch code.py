def varied_basis_set(num_gto):
    from pyqint import cgf

    basis_set = cgf([0.0, 0.0, 0.0])

    c = 1/num_gto
    alpha = 1

    for i in range(num_gto):
        cgf.add_gto(basis_set, c, alpha, 0, 0, 0)

    return basis_set

print(f'varied basis set: {varied_basis_set(3)}')