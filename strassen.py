import numpy as np
import csv
import time
import sys
import random 

FLAG = int(sys.argv[1])
DIMENSION = int(sys.argv[2])
FILE = sys.argv[3]

CUTOFF = 512

def main():
    file = open(FILE, "r")

    X = np.empty([DIMENSION, DIMENSION], dtype=np.int32)
    Y = np.empty([DIMENSION, DIMENSION], dtype=np.int32)

    for r in range(DIMENSION):
        for c in range(DIMENSION):
            X[r][c] = file.readline()
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            Y[r][c] = file.readline()

    Z = strassen(X, Y, CUTOFF)
    for i in range(DIMENSION):
        print(Z[i][i])

def conventional(X, Y):
    n = np.shape(X)[0]
    prod = np.empty([n, n], dtype=np.int32)
    for r in range(n):
        for c in range(n):
            prod[r][c] = np.dot(X[r, :], Y[:, c])
    return prod

def strassen(X, Y, cutoff):
    n = np.shape(X)[0]

    if n <= cutoff:
       return conventional(X, Y)

    prod = np.empty([n, n], dtype=np.int32)

    # Pad with zeros if odd
    if n % 2:
        X = np.pad(X, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        Y = np.pad(Y, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        padded_prod = strassen(X, Y, cutoff)
        prod = padded_prod[:n, :n]

    else:
        mid = n // 2

        A = X[:mid, :mid]
        B = X[:mid, mid:]
        C = X[mid:, :mid]
        D = X[mid:, mid:]
        E = Y[:mid, :mid]
        F = Y[:mid, mid:]
        G = Y[mid:, :mid]
        H = Y[mid:, mid:]

        P1 = strassen(A, F - H, cutoff)
        P2 = strassen(A + B, H, cutoff)
        P3 = strassen(C + D, E, cutoff)
        P4 = strassen(D, G - E, cutoff)
        P5 = strassen(A + D, E + H, cutoff)
        P6 = strassen(B - D, G + H, cutoff)
        P7 = strassen(C - A, E + F, cutoff)

        Q1 = -P2 + P4 + P5 + P6
        Q2 = P1 + P2 
        Q3 = P3 + P4
        Q4 = P1 - P3 + P5 + P7

        top = np.concatenate((Q1, Q2), axis=1)
        bottom = np.concatenate((Q3, Q4), axis=1)
        prod = np.concatenate((top, bottom), axis=0)

    return prod

def get_times():
    fields = ['cutoff', 'size', 'time']
    filename = 'times.csv'
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

        for N in [1024, 2048]:
            rng = np.random.default_rng()
            X = rng.integers(-1, 3, size=(N, N))
            Y = rng.integers(-1, 3, size=(N, N))

            for n0 in (64, 128, 256, 512, 1024):
                start = time.time()
                strassen(X, Y, n0)
                end = time.time()
                csvwriter.writerow([n0, N, end - start])
                print([n0, N, end - start])

def tests():
    rng = np.random.default_rng()
    for N in range (1,6):
        for low in range(-1, 1):
            for high in range(1, 3):
                X1 = rng.integers(low, high + 1, size=(N, N))
                Y1 = rng.integers(low, high + 1, size=(N, N))
                Z1 = strassen(X1, Y1, CUTOFF)
                if (Z1==np.matmul(X1, Y1, dtype=np.int32)).all():
                    print(f"Test passed: {N}x{N}, {low} to {high}")
                else:
                    print(f"Test failed: {N}x{N}, {low} to {high}")

def triangles(A):
    A_2 = strassen(A, A, CUTOFF)
    A_3 = strassen(A_2, A, CUTOFF)
    
    n = A_3.shape[0] # can also say len(A_3)

    sum = 0
    for i in range(n):
        sum += A_3[i][i]
    
    return sum / 6

def make_graph(p):
    vertices = 1024
    adj_matrix = np.empty((vertices, vertices))
    for i in range(vertices): 
        for j in range(vertices):
            r = random.random()
            if i == j: 
                adj_matrix[i][j] = 0
            else:
                if r <= p:
                    adj_matrix[i][j] = 1
                else: 
                    adj_matrix[i][j] = 0
    return adj_matrix

def get_triangles():
    for p in [0.01, 0.02, 0.03, 0.04, 0.05]: 
        A = make_graph(p)
        num_triangles = triangles(A)
        print(f"Number of triangles for p = {p} is {num_triangles} ")

main()