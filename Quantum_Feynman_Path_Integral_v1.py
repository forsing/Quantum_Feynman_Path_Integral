# Quantum_Feynman_Path_Integral
# Richard Feynman



import csv
import math
import random
from itertools import combinations, islice
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector


random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91_loto_2982.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91_loto_plus_1738.csv"),
)



MODES = 39
PARTICLES = 7
REGISTER_DIM = 64
QUBITS_PER_REGISTER = 6

TRAIN_BATCH = 256
SEARCH_BATCH = 4096

# Veci model zahteva proveru jace regularizacije.
REG_GRID = (0.01, 0.1, 1.0)
GRAD_TOL = 0.001

OPTIMIZATION_STAGES = (
    ("BFGS", 300),
    ("CG", 400),
    ("BFGS", 400),
)

PARAM_COUNT = MODES * MODES


class ConvergenceError(RuntimeError):
    pass


def load_draws(path):
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle), 1):
            if not fields or all(not field.strip() for field in fields):
                continue

            if len(fields) != PARTICLES:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "potrebno je tacno sedam kolona bez hedera."
                )

            try:
                numbers = [int(field.strip()) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from exc

            if (
                len(set(numbers)) != PARTICLES
                or min(numbers) < 1
                or max(numbers) > MODES
            ):
                raise ValueError(
                    f"{path.name}, red {line_number}: nelegalna kombinacija."
                )

            rows.append(sorted(numbers))

    if len(rows) < 80:
        raise ValueError(f"{path.name}: nedovoljno istorijskih kola.")

    # Hronologija kola se ne menja.
    return np.asarray(rows, dtype=np.int64) - 1


def initial_parameters():
    """
    Fiksna, neslucajna inicijalizacija svih parnih veza.
    Nema izdvajanja razmaka 1, 2, 3 ili drugih susedstava.
    """
    row = np.arange(1, MODES + 1, dtype=float)[:, None]
    column = np.arange(1, MODES + 1, dtype=float)[None, :]

    matrix = (
        2.5
        / np.sqrt(MODES)
        * np.sin(
            np.sqrt(2.0) * row
            + np.sqrt(3.0) * column
            + np.sqrt(5.0) * row * column
        )
    )

    matrix -= np.eye(MODES) * np.trace(matrix) / MODES
    return matrix.ravel()


def hamiltonian(params):
    """
    Svaki realni A odgovara hermitskom H:

    H = (A + A.T)/2 + i(A - A.T)/2.

    Uklanja se trag, jer globalna faza ne menja verovatnoce.
    Nema fiksnog grafa, ciklusa ili maske susednih brojeva.
    """
    matrix = np.asarray(params).reshape(MODES, MODES)

    result = (
        0.5 * (matrix + matrix.T)
        + 0.5j * (matrix - matrix.T)
    )

    result -= np.eye(MODES) * np.trace(result) / MODES
    return result


def spectral_propagator(params):
    h = hamiltonian(params)
    values, vectors = np.linalg.eigh(h)

    unitary = (
        vectors * np.exp(-1j * values)[None, :]
    ) @ vectors.conj().T

    return h, values, vectors, unitary


def propagator(params):
    return spectral_propagator(params)[3]


def check_unitary(unitary):
    if not np.allclose(
        unitary.conj().T @ unitary,
        np.eye(len(unitary)),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise FloatingPointError("Propagator nije unitaran.")


def qiskit_circuit(unitary):
    embedded = np.eye(REGISTER_DIM, dtype=complex)
    embedded[1:MODES + 1, 1:MODES + 1] = unitary

    gate = UnitaryGate(embedded, label="exp(-iH)")
    registers = [
        QuantumRegister(QUBITS_PER_REGISTER, f"r{index + 1}")
        for index in range(PARTICLES)
    ]

    circuit = QuantumCircuit(*registers)

    for register in registers:
        circuit.append(gate, list(register))

    # Ovo gradi propagaciju registara.
    # Antisimetricno pocetno stanje racuna se u sazetom,
    # Slaterovom obliku, bez punog 42-qubitnog statevectora.
    return circuit, gate, embedded


def objective(params, draws, start, end, regularization, gradient=True):
    """
    Koristi stvarne prelaze draws[t-1] -> draws[t].

    P(y | x) = |det(U[y, x])|^2.
    """
    count = end - start

    if start < 1 or count <= 0:
        raise ValueError("Neispravan opseg podataka.")

    h, eigenvalues, eigenvectors, unitary = spectral_propagator(params)
    loss = 0.0
    unitary_gradient = np.zeros_like(unitary)

    for first in range(start, end, TRAIN_BATCH):
        last = min(first + TRAIN_BATCH, end)
        previous = draws[first - 1:last - 1]
        targets = draws[first:last]

        rows = targets[:, :, None]
        columns = previous[:, None, :]
        minors = unitary[rows, columns]

        signs, log_absolute = np.linalg.slogdet(minors)

        if (
            np.any(np.abs(signs) == 0.0)
            or not np.all(np.isfinite(log_absolute))
        ):
            if gradient:
                return np.inf, np.zeros(PARAM_COUNT)
            return np.inf

        loss -= 2.0 * float(log_absolute.sum())

        if gradient:
            inverse = np.linalg.inv(minors)

            # Konvencija:
            # dL = Re(sum(conj(G_U) * dU)).
            contributions = -2.0 * inverse.conj().transpose(0, 2, 1)

            broadcast_rows = np.broadcast_to(rows, minors.shape)
            broadcast_columns = np.broadcast_to(columns, minors.shape)

            np.add.at(
                unitary_gradient,
                (
                    broadcast_rows.ravel(),
                    broadcast_columns.ravel(),
                ),
                contributions.ravel(),
            )

    loss /= count
    loss += regularization * float(np.vdot(h, h).real) / MODES

    if not gradient:
        return loss

    unitary_gradient /= count

    # Adjungovani izvod matricnog eksponencijala.
    difference = eigenvalues[:, None] - eigenvalues[None, :]
    midpoint = (
        eigenvalues[:, None] + eigenvalues[None, :]
    ) / 2.0

    divided_difference = (
        -1j
        * np.exp(-1j * midpoint)
        * np.sinc(difference / (2.0 * np.pi))
    )

    transformed_gradient = (
        eigenvectors.conj().T
        @ unitary_gradient
        @ eigenvectors
    )

    h_gradient = (
        eigenvectors
        @ (divided_difference.conj() * transformed_gradient)
        @ eigenvectors.conj().T
    )

    h_gradient += 2.0 * regularization * h / MODES

    # Povratak sa H na realne parametre A.
    real_part = h_gradient.real
    imaginary_part = h_gradient.imag

    matrix_gradient = (
        0.5 * (real_part + real_part.T)
        + 0.5 * (imaginary_part - imaginary_part.T)
    )

    matrix_gradient -= (
        np.eye(MODES) * np.trace(real_part) / MODES
    )

    result_gradient = matrix_gradient.ravel()

    if (
        not np.isfinite(loss)
        or not np.all(np.isfinite(result_gradient))
    ):
        raise FloatingPointError("Nefinitan cilj ili gradijent.")

    return loss, result_gradient


def fit(draws, end, regularization, initial=None, label="ucenje"):
    params = (
        initial_parameters()
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )

    best_x = None
    best_value = np.inf
    best_gradient = None

    cached_x = None
    cached_value = None
    cached_gradient = None

    def function(values):
        nonlocal best_x, best_value, best_gradient
        nonlocal cached_x, cached_value, cached_gradient

        values = np.asarray(values, dtype=float)

        if cached_x is not None and np.array_equal(values, cached_x):
            return cached_value, cached_gradient.copy()

        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Nefinitni parametri.")

        value, derivative = objective(
            values, draws, 1, end, regularization, True
        )

        cached_x = values.copy()
        cached_value = float(value)
        cached_gradient = derivative.copy()

        if np.isfinite(value):
            better = value < best_value

            if best_gradient is not None and value == best_value:
                better = (
                    np.max(np.abs(derivative))
                    < np.max(np.abs(best_gradient))
                )

            if better:
                best_x = values.copy()
                best_value = float(value)
                best_gradient = derivative.copy()

        return float(value), derivative.copy()

    initial_value, initial_gradient = function(params)

    if not np.isfinite(initial_value):
        raise FloatingPointError(
            f"{label}: pocetni model daje nulti posmatrani prelaz."
        )

    if np.max(np.abs(initial_gradient)) <= GRAD_TOL:
        return params

    reports = []

    for stage, (method, limit) in enumerate(OPTIMIZATION_STAGES):
        if stage == 1:
            print(
                f"Nastavak optimizacije: {label}; "
                f"gradijent={np.max(np.abs(best_gradient)):.6g}.",
                flush=True,
            )

        result = minimize(
            function,
            best_x.copy(),
            jac=True,
            method=method,
            options={
                "maxiter": limit,
                "gtol": GRAD_TOL,
                "norm": np.inf,
            },
        )

        value, derivative = function(result.x)
        norm = float(np.max(np.abs(derivative)))

        reports.append(
            f"{method}: status={result.status}, "
            f"iteracije={result.nit}, gradijent={norm:.6g}, "
            f"razlog={str(result.message).strip()}"
        )

        if (
            np.isfinite(value)
            and norm <= GRAD_TOL
            and value <= initial_value + 1e-10
        ):
            return np.asarray(result.x).copy()

        if (
            np.max(np.abs(best_gradient)) <= GRAD_TOL
            and best_value <= initial_value + 1e-10
        ):
            return best_x.copy()

    raise ConvergenceError(
        f"{label}: nije dostignuta konvergencija. "
        + " | ".join(reports)
    )


def enumerate_distribution(orbitals, particles, batch_size):
    modes = orbitals.shape[0]

    if orbitals.shape[1] != particles:
        raise ValueError("Neispravan broj orbitala.")

    if not np.allclose(
        orbitals.conj().T @ orbitals,
        np.eye(particles),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise FloatingPointError("Orbitale nisu ortonormirane.")

    iterator = combinations(range(modes), particles)
    best_log_probability = -np.inf
    best_combination = None
    masses = []
    visited = 0

    while True:
        chunk = list(islice(iterator, batch_size))

        if not chunk:
            break

        indices = np.asarray(chunk, dtype=np.int64)
        minors = orbitals[indices, :]
        _, log_absolute = np.linalg.slogdet(minors)

        if np.any(np.isnan(log_absolute)):
            raise FloatingPointError("Neispravna determinanta.")

        log_probabilities = 2.0 * log_absolute
        index = int(np.argmax(log_probabilities))
        score = float(log_probabilities[index])

        if score > best_log_probability:
            best_log_probability = score
            best_combination = tuple(
                int(number) for number in indices[index]
            )

        masses.append(float(np.exp(log_probabilities).sum()))
        visited += len(chunk)

    if visited != math.comb(modes, particles):
        raise RuntimeError("Pretraga nije obuhvatila sve kombinacije.")

    probability_mass = math.fsum(masses)

    if not np.isclose(
        probability_mass, 1.0, atol=1e-8, rtol=1e-8
    ):
        raise FloatingPointError(
            f"Raspodela nije normalizovana: {probability_mass:.12g}."
        )

    if best_combination is None:
        raise RuntimeError("Nema kombinacije pozitivne mase.")

    return best_combination, best_log_probability


def self_check():
    params = initial_parameters()
    h = hamiltonian(params)

    if not np.allclose(h, h.conj().T, atol=1e-12):
        raise RuntimeError("Hamiltonijan nije hermitski.")

    if not np.isclose(np.trace(h), 0.0, atol=1e-12):
        raise RuntimeError("Trag Hamiltonijana nije uklonjen.")

    unitary = propagator(params)
    check_unitary(unitary)

    circuit, gate, embedded = qiskit_circuit(unitary)

    if circuit.num_qubits != PARTICLES * QUBITS_PER_REGISTER:
        raise RuntimeError("Neispravan broj registara.")

    if not np.allclose(
        Operator(gate).data, embedded, atol=1e-12
    ):
        raise RuntimeError("Qiskit operator nije ispravan.")

    for number in (1, 19, 39):
        state = Statevector.from_int(
            number, dims=REGISTER_DIM
        ).evolve(gate)

        if not np.allclose(
            state.data, embedded[:, number], atol=1e-12
        ):
            raise RuntimeError("Qiskit statevector provera nije prosla.")

    # Mala iscrpna provera normalizacije i maksimuma.
    small_modes = 6
    indices = np.arange(small_modes)
    small_unitary = np.exp(
        2j * np.pi * indices[:, None] * indices[None, :] / small_modes
    ) / np.sqrt(small_modes)

    for particles in (1, 3, 6):
        orbitals = small_unitary[:, :particles]
        selected, score = enumerate_distribution(
            orbitals, particles, 4
        )

        exact_maximum = max(
            abs(np.linalg.det(orbitals[list(subset), :])) ** 2
            for subset in combinations(range(small_modes), particles)
        )

        selected_probability = abs(
            np.linalg.det(orbitals[list(selected), :])
        ) ** 2

        if not np.isclose(
            np.exp(score), exact_maximum, atol=1e-12, rtol=1e-10
        ):
            raise RuntimeError("Provera maksimuma nije prosla.")

        if not np.isclose(
            selected_probability, exact_maximum, atol=1e-12, rtol=1e-10
        ):
            raise RuntimeError("Izabrana kombinacija nije maksimum.")

    sample = np.asarray(
        [
            [1, 4, 9, 16, 22, 31, 38],
            [2, 7, 11, 18, 25, 30, 39],
            [3, 5, 12, 20, 24, 32, 37],
            [4, 8, 14, 19, 27, 33, 36],
        ],
        dtype=np.int64,
    ) - 1

    value, analytical = objective(
        params, sample, 1, len(sample), 0.1, True
    )

    # Provera gradijenta u vise fiksnih pravaca i koordinata.
    # Nema slucajnih testnih pravaca.
    directions = []

    for index in (0, 1, 38, 39, 147, 759, 1520):
        direction = np.zeros(PARAM_COUNT)
        direction[index] = 1.0
        directions.append(direction)

    coordinates = np.arange(1, PARAM_COUNT + 1, dtype=float)

    for frequency in (0.37, 0.83, 1.41):
        direction = np.sin(frequency * coordinates)
        direction /= np.linalg.norm(direction)
        directions.append(direction)

    step = 1e-6

    for direction in directions:
        upper = objective(
            params + step * direction,
            sample,
            1,
            len(sample),
            0.1,
            False,
        )

        lower = objective(
            params - step * direction,
            sample,
            1,
            len(sample),
            0.1,
            False,
        )

        numerical = (upper - lower) / (2.0 * step)
        predicted = float(np.dot(analytical, direction))

        if not np.isclose(
            numerical, predicted, atol=2e-5, rtol=2e-4
        ):
            raise RuntimeError(
                "Provera gradijenta nije prosla: "
                f"analiticki={predicted:.9g}, numericki={numerical:.9g}."
            )

    # Model i regularizacija ne zavise od numerickog susedstva.
    # Istovremeno preimenovanje modova i podataka mora cuvati cilj.
    permutation = (11 * np.arange(MODES)) % MODES
    inverse_permutation = np.argsort(permutation)

    permuted_h = h[np.ix_(permutation, permutation)]
    permuted_params = (
        permuted_h.real + permuted_h.imag
    ).ravel()

    permuted_draws = np.sort(inverse_permutation[sample], axis=1)

    permuted_value = objective(
        permuted_params,
        permuted_draws,
        1,
        len(sample),
        0.1,
        False,
    )

    if not np.isclose(
        value, permuted_value, atol=1e-8, rtol=1e-8
    ):
        raise RuntimeError(
            "Provera nezavisnosti modela od oznaka brojeva nije prosla."
        )


def longest_run(sequence):
    current = 1
    longest = 1

    for left, right in zip(sequence, sequence[1:]):
        current = current + 1 if right == left + 1 else 1
        longest = max(longest, current)

    return longest


def process_csv(path):
    draws = load_draws(path)
    transition_count = len(draws) - 1

    train_end = 1 + int(0.70 * transition_count)
    validation_end = 1 + int(0.85 * transition_count)

    best = None

    for regularization in REG_GRID:
        try:
            parameters = fit(
                draws,
                train_end,
                regularization,
                label=f"trening reg={regularization:g}",
            )
        except ConvergenceError as exc:
            print(f"PROVERA: {exc}", flush=True)
            continue

        validation = objective(
            parameters,
            draws,
            train_end,
            validation_end,
            0.0,
            False,
        )

        if np.isfinite(validation):
            if best is None or validation < best[0]:
                best = (validation, regularization, parameters)

    if best is None:
        raise ConvergenceError(
            "Nema konvergiranog kandidata sa konacnom validacionom NLL."
        )

    validation, regularization, train_parameters = best

    test_parameters = fit(
        draws,
        validation_end,
        regularization,
        initial=train_parameters,
        label="ucenje trening + validacija",
    )

    test_nll = objective(
        test_parameters,
        draws,
        validation_end,
        len(draws),
        0.0,
        False,
    )

    full_parameters = fit(
        draws,
        len(draws),
        regularization,
        initial=test_parameters,
        label="zavrsno ucenje na celoj istoriji",
    )

    unitary = propagator(full_parameters)
    check_unitary(unitary)

    # Fizicke orbitale sedam zauzetih ulaznih modova.
    orbitals = unitary[:, draws[-1]]

    print(
        f"Validacija NLL: {validation:.6f} | reg: {regularization:g}",
        flush=True,
    )
    print(f"Test NLL: {test_nll:.6f}", flush=True)
    print(
        f"Pretraga NEXT: svih {math.comb(MODES, PARTICLES)} kombinacija.",
        flush=True,
    )

    selected, log_probability = enumerate_distribution(
        orbitals, PARTICLES, SEARCH_BATCH
    )

    next_combination = tuple(number + 1 for number in selected)
    print(f"NEXT: {list(next_combination)}", flush=True)

    if longest_run(next_combination) >= 5:
        print(
            "PROVERA: najmanje pet susednih brojeva; "
            "rezultat zahteva proveru.",
            flush=True,
        )

    if not np.isfinite(test_nll):
        print(
            "PROVERA: najmanje jedan test prelaz ima nultu verovatnocu.",
            flush=True,
        )

    reference = np.sum(np.abs(orbitals) ** 2, axis=1)
    differences = []

    for index in (0, len(draws) // 2):
        earlier = np.sum(
            np.abs(unitary[:, draws[index]]) ** 2,
            axis=1,
        )

        differences.append(
            float(np.max(np.abs(reference - earlier)))
        )

    if max(differences) < 1e-8:
        print(
            "PROVERA: slaba osetljivost marginala "
            "na proverene ulazne kombinacije.",
            flush=True,
        )

    return {
        "path": path,
        "next": next_combination,
        "parameters": full_parameters,
        "log_probability": log_probability,
    }


def main():
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(str(path))

    print(
        "Quantum Feynman Path Integral: "
        "uceni Hamiltonijan svih parnih veza.",
        flush=True,
    )
    print(
        "Sedam registara po sest qubita; "
        "lokalna simulacija fermionskih amplituda.",
        flush=True,
    )

    self_check()
    print("Interne provere: prosle.", flush=True)

    results = []

    for path in FILES:
        print(f"CSV: {path.name}", flush=True)

        try:
            results.append(process_csv(path))
        except ConvergenceError as exc:
            print(f"STOP za ovaj CSV: {exc}", flush=True)

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            left = results[first]
            right = results[second]

            if left["next"] == right["next"]:
                same_parameters = np.allclose(
                    left["parameters"],
                    right["parameters"],
                    atol=1e-10,
                    rtol=1e-8,
                )

                print(
                    "PROVERA: ista NEXT kombinacija za "
                    f"{left['path'].name} i {right['path'].name}.",
                    flush=True,
                )
                print(
                    "Parametri modela: "
                    + (
                        "prakticno isti."
                        if same_parameters
                        else "razliciti."
                    ),
                    flush=True,
                )

    if len(results) != len(FILES):
        print(
            "Obrada nije kompletna: neki modeli nisu konvergirali.",
            flush=True,
        )


if __name__ == "__main__":
    main()



"""
Quantum Feynman Path Integral: uceni Hamiltonijan svih parnih veza.
Sedam registara po sest qubita; lokalna simulacija fermionskih amplituda.
Interne provere: prosle.

CSV: loto7_4720_k91.csv
Validacija NLL: 17.583927 | reg: 0.01
Test NLL: 17.629388
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [7, 12, 16, 17, 20, 25, 29]

CSV: loto7_4720_k91_loto_2982.csv
Nastavak optimizacije: trening reg=0.1; gradijent=0.00141371.
Validacija NLL: 17.616487 | reg: 0.1
Test NLL: 17.619598
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [13, 15, 17, 22, 23, 25, 27]

CSV: loto7_4720_k91_loto_plus_1738.csv
Validacija NLL: 17.636470 | reg: 0.01
Test NLL: 17.922452
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [5, 12, 14, 19, 34, 35, 36]
"""



"""
Propagator čije faze i interakcije zavise od istorijskih podataka, pa raspodelu sledeće kombinacije dobijam interferencijom alternativnih prelaza. 
Propagator sedam neinteragujućih fermiona. 
Amplituda cele kombinacije računa se determinantom, koja uključuje interferenciju alternativnih prelaza. 
To je ograničen kvantni model, ne opšti model svih mogućih interakcija. 
Sedam registara od po šest qubita simulira se u sažetom obliku, bez ogromnog vektora za 42 qubita. 
Klasični optimizator uči parametre Hamiltonijana. 
Nema klasičnog izlaznog regresora.
Završni izbor proverava svih 15.380.937 kombinacija po CSV-u, u malim paketima; taj deo može potrajati.
Fajnmanova suma po putanjama i interferencija amplituda. 
Zasebno se uče sve parne veze u Hamiltonijanu, bez privilegovanja susednih brojeva. 
Dodao sam odgovarajuću regularizacija, izabranu hronološkom validacijom.
Fajnmanov propagator, interferencija i iscrpna pretraga. 
Nema filtera koji zabranjuje uzastopne brojeve. 
"""
