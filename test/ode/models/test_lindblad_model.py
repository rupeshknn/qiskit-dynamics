# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""tests for quantum_models.LindbladModel"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info.operators import Operator
from qiskit_ode.models.quantum_models import HamiltonianModel, LindbladModel
from qiskit_ode.models.signals import Constant, Signal, VectorSignal


class TestLindbladModel(unittest.TestCase):
    """Tests for LindbladModel.
    """

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

        # define a basic hamiltonian
        w = 2.
        r = 0.5
        ham_operators = [2 * np.pi * self.Z / 2,
                     2 * np.pi * r * self.X / 2]
        ham_signals = [Constant(w), Signal(1., w)]

        self.w = w
        self.r = r

        noise_operators = np.array([[[0., 0.], [1., 0.]]])

        self.basic_lindblad = LindbladModel(hamiltonian_operators=ham_operators,
                                            hamiltonian_signals=ham_signals,
                                            noise_operators=noise_operators)


    def test_basic_lindblad_lmult(self):
        """Test lmult method of Lindblad generator OperatorModel.
        """
        A = np.array([[1., 2.], [3., 4.]])

        t = 1.123
        ham = (2 * np.pi * self.w * self.Z.data / 2
               + 2 * np.pi * self.r * np.cos(2 * np.pi * self.w * t) * self.X.data / 2)
        sm = np.array([[0., 0.], [1., 0.]])
        sp = sm.transpose()

        expected = self._evaluate_lindblad_rhs(A, ham, [sm])
        value = self.basic_lindblad.lmult(t, A.flatten(order='F'))
        self.assertAlmostEqual(expected, value.reshape(2,2, order='F'))


    def test_lindblad_lmult_pseudorandom(self):
        """Test lmult of Lindblad OperatorModel with structureless
        pseudorandom model parameters.
        """
        rng = np.random.default_rng(9848)
        dim = 10
        num_ham = 4
        num_diss = 3

        b = 1. # bound on size of random terms

        # generate random hamiltonian
        rand_operators = (rng.uniform(low=-b,high=b, size=(num_ham, dim, dim)) +
                          1j * rng.uniform(low=-b,high=b, size=(num_ham, dim, dim)))
        rand_ham_ops = rand_operators + rand_operators.conj().transpose([0, 2, 1])

        # generate random hamiltonian coefficients
        rand_ham_coeffs = (rng.uniform(low=-b,high=b, size=(num_ham)) +
                           1j * rng.uniform(low=-b,high=b, size=(num_ham)))
        rand_ham_carriers = rng.uniform(low=-b,high=b, size=(num_ham))
        ham_sigs = VectorSignal(lambda t: rand_ham_coeffs, rand_ham_carriers)

        # generate random dissipators
        rand_diss = (rng.uniform(low=-b,high=b, size=(num_diss, dim, dim)) +
                     1j * rng.uniform(low=-b,high=b, size=(num_diss, dim, dim)))

        # random dissipator coefficients
        rand_diss_coeffs = (rng.uniform(low=-b,high=b, size=(num_diss)) +
                           1j * rng.uniform(low=-b,high=b, size=(num_diss)))
        rand_diss_carriers = rng.uniform(low=-b,high=b, size=(num_diss))
        diss_sigs = VectorSignal(lambda t: rand_diss_coeffs, rand_diss_carriers)


        # random anti-hermitian frame operator
        rand_op = (rng.uniform(low=-b, high=b, size=(dim,dim)) +
                   1j*rng.uniform(low=-b, high=b, size=(dim,dim)))
        frame_op = rand_op - rand_op.conj().transpose()

        lindblad_frame_op = (np.kron(np.eye(dim), frame_op)
                             - np.kron(frame_op.transpose(), np.eye(dim)))

        # construct model
        hamiltonian = HamiltonianModel(operators=rand_ham_ops,
                                       signals=ham_sigs)
        lindblad_model = LindbladModel.from_hamiltonian(hamiltonian=hamiltonian,
                                                 noise_operators=rand_diss,
                                                 noise_signals=diss_sigs)
        lindblad_model.frame = lindblad_frame_op

        A = (rng.uniform(low=-b,high=b, size=(dim, dim)) +
             1j * rng.uniform(low=-b,high=b, size=(dim, dim)))

        t = rng.uniform(low=-b, high=b)
        value = lindblad_model.lmult(t, A.flatten(order='F'))

        ham_coeffs = np.real(rand_ham_coeffs * np.exp(1j * 2 * np.pi * rand_ham_carriers * t))
        ham = np.tensordot(ham_coeffs, rand_ham_ops, axes=1)
        diss_coeffs = np.real(rand_diss_coeffs * np.exp(1j * 2 * np.pi * rand_diss_carriers * t))

        expected = self._evaluate_lindblad_rhs(A, ham,
                                               rand_diss,
                                               diss_coeffs,
                                               frame_op,
                                               t)

        self.assertAlmostEqual(expected, value.reshape(dim,dim, order='F'))

    def _evaluate_lindblad_rhs(self, A, ham,
                               dissipators=None,
                               dissipator_coeffs=None,
                               frame_op=None,
                               t=0.):
        """Evaluate the Lindblad equation

        frame_op assumed anti-Hermitian
        """
        # if a frame operator is given, transform the model pieces into
        # the frame
        if frame_op is not None:
            U = expm(-t * frame_op)
            Uadj = U.conj().transpose()

            ham = U @ ham @ Uadj - 1j * frame_op

            if dissipators is not None:
                dissipators = [U @ D @ Uadj for D in dissipators]

        ham_part = -1j * (ham @ A - A @ ham)

        if dissipators is None:
            return ham_part

        if dissipator_coeffs is None:
            dissipator_coeffs = np.ones(len(dissipators))

        diss_part = np.zeros_like(A)
        for c, D in zip(dissipator_coeffs, dissipators):
            Dadj = D.conj().transpose()
            DadjD = Dadj @ D
            diss_part += c * (D @ A @ Dadj - 0.5 * (DadjD @ A + A @ DadjD))

        return ham_part + diss_part


    def assertAlmostEqual(self, A, B, tol=1e-12):
        self.assertTrue(np.abs(A - B).max() < tol)