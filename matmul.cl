__kernel void parallel_mult_mat(const unsigned long M, const unsigned long N, const unsigned long K,
                      const __global double* A,
                      const __global double* B,
                      __global double* C) {
    
    const unsigned long globalRow = get_global_id(0); // Row ID of C (0..M)
    const unsigned long globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    double acc = 0.0f;
    for (unsigned long k=0; k<K; k++) {
        acc += B[k*M + globalRow] * A[globalCol*K + k];
    }
 
    // Store the result
    C[globalCol*M + globalRow] = acc;
}