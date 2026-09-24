//! Native Clarabel backend for the powered-descent landing planner (ABI 2).
//!
//! The landing SOCP is assembled in Python (`guidance.ConvexLandingPlanner`),
//! so the Rust and Python paths always solve the *same* matrices.  This DLL
//! is a thin, allocation-safe C ABI around Clarabel 0.11:
//!
//! ```text
//! minimise   1/2 x'Px + q'x
//! subject to A x + s = b,   s in  {0}^z  x  R+^l  x  SOC(d1) x ... x SOC(dk)
//! ```
//!
//! `P` must be upper triangular; `P` and `A` are passed in CSC form with
//! `usize` (= `size_t`) indices.  The historic `clarabel_hover_*` symbol
//! names are kept so the DLL file name and loader stay the same.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};
use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

const ABI_VERSION: u32 = 2;
const BACKEND_NAME: &[u8] = b"clarabel-rust-conic-0.11.1\0";

#[repr(C)]
pub struct HoverDiagnostics {
    pub status: i32,
    pub iterations: u32,
    pub objective: f64,
    pub solve_time_seconds: f64,
}

#[no_mangle]
pub extern "C" fn clarabel_hover_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn clarabel_hover_backend_name() -> *const c_char {
    BACKEND_NAME.as_ptr().cast()
}

/// Validate one CSC matrix supplied through the C ABI and copy it.
///
/// # Safety
/// The pointers must reference `ncols + 1` column pointers and `nnz` row
/// indices / values, as documented for [`clarabel_conic_solve`].
unsafe fn read_csc(
    nrows: usize,
    ncols: usize,
    colptr: *const usize,
    rowval: *const usize,
    nzval: *const f64,
    nnz: usize,
    upper_triangular: bool,
) -> Result<CscMatrix<f64>, i32> {
    if colptr.is_null() || (nnz > 0 && (rowval.is_null() || nzval.is_null())) {
        return Err(-1);
    }
    let colptr = slice::from_raw_parts(colptr, ncols + 1).to_vec();
    let (rowval, nzval) = if nnz > 0 {
        (
            slice::from_raw_parts(rowval, nnz).to_vec(),
            slice::from_raw_parts(nzval, nnz).to_vec(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    if colptr[0] != 0 || colptr[ncols] != nnz {
        return Err(-10);
    }
    for column in 0..ncols {
        let (start, end) = (colptr[column], colptr[column + 1]);
        if start > end || end > nnz {
            return Err(-11);
        }
        let mut previous: Option<usize> = None;
        for index in start..end {
            let row = rowval[index];
            if row >= nrows || previous.is_some_and(|p| p >= row) {
                return Err(-12); // out of range or not strictly sorted
            }
            if upper_triangular && row > column {
                return Err(-13);
            }
            if !nzval[index].is_finite() {
                return Err(-3);
            }
            previous = Some(row);
        }
    }
    Ok(CscMatrix::new(nrows, ncols, colptr, rowval, nzval))
}

/// Solve one conic program and write the primal solution to `x_out`.
///
/// Returns the Clarabel status discriminant (`1` Solved, `4` AlmostSolved,
/// other positive values for the remaining statuses).  Negative values are
/// ABI/validation errors; nothing is written to `x_out` in that case.
///
/// # Safety
/// All pointers must be valid for the lengths given by the other arguments:
/// `p_colptr`/`a_colptr` hold `n + 1` entries, `p_rowval`/`p_nzval` hold
/// `p_nnz`, `a_rowval`/`a_nzval` hold `a_nnz`, `q` and `x_out` hold `n`,
/// `b` holds `m`, and `soc_dims` holds `soc_count` entries.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn clarabel_conic_solve(
    n: usize,
    m: usize,
    p_colptr: *const usize,
    p_rowval: *const usize,
    p_nzval: *const f64,
    p_nnz: usize,
    q: *const f64,
    a_colptr: *const usize,
    a_rowval: *const usize,
    a_nzval: *const f64,
    a_nnz: usize,
    b: *const f64,
    zero_rows: usize,
    nonnegative_rows: usize,
    soc_dims: *const usize,
    soc_count: usize,
    max_iter: u32,
    x_out: *mut f64,
    x_out_len: usize,
    diagnostics_out: *mut HoverDiagnostics,
) -> i32 {
    if q.is_null() || b.is_null() || x_out.is_null() || diagnostics_out.is_null() {
        return -1;
    }
    if (soc_count > 0 && soc_dims.is_null()) || x_out_len < n || n == 0 {
        return -2;
    }

    let result = catch_unwind(AssertUnwindSafe(|| -> i32 {
        let p = match read_csc(n, n, p_colptr, p_rowval, p_nzval, p_nnz, true) {
            Ok(matrix) => matrix,
            Err(code) => return code,
        };
        let a = match read_csc(m, n, a_colptr, a_rowval, a_nzval, a_nnz, false) {
            Ok(matrix) => matrix,
            Err(code) => return code,
        };
        let q = slice::from_raw_parts(q, n);
        let b = slice::from_raw_parts(b, m);
        if !q.iter().chain(b.iter()).all(|value| value.is_finite()) {
            return -3;
        }

        let socs: &[usize] = if soc_count > 0 {
            slice::from_raw_parts(soc_dims, soc_count)
        } else {
            &[]
        };
        if socs.iter().any(|&dim| dim < 1)
            || zero_rows + nonnegative_rows + socs.iter().sum::<usize>() != m
        {
            return -14;
        }
        let mut cones = Vec::<SupportedConeT<f64>>::with_capacity(socs.len() + 2);
        if zero_rows > 0 {
            cones.push(SupportedConeT::ZeroConeT(zero_rows));
        }
        if nonnegative_rows > 0 {
            cones.push(SupportedConeT::NonnegativeConeT(nonnegative_rows));
        }
        cones.extend(socs.iter().map(|&dim| SupportedConeT::SecondOrderConeT(dim)));

        let settings = match DefaultSettingsBuilder::default()
            .verbose(false)
            .max_iter(max_iter.max(1))
            .build()
        {
            Ok(settings) => settings,
            Err(_) => return -6,
        };
        let mut solver = match DefaultSolver::new(&p, q, &a, b, &cones, settings) {
            Ok(solver) => solver,
            Err(_) => return -4,
        };
        solver.solve();

        let status = solver.solution.status;
        let status_code = status as i32;
        diagnostics_out.write(HoverDiagnostics {
            status: status_code,
            iterations: solver.solution.iterations,
            objective: solver.solution.obj_val,
            solve_time_seconds: solver.solution.solve_time,
        });
        if matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved)
            && solver.solution.x.len() == n
            && solver.solution.x.iter().all(|value| value.is_finite())
        {
            slice::from_raw_parts_mut(x_out, n).copy_from_slice(&solver.solution.x);
        }
        status_code
    }));

    result.unwrap_or(-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// min (x0-3)^2 + (x1-4)^2 + t   s.t.  x0 + x1 = 1.5,  t <= 2,  ||x|| <= t
    #[test]
    fn solves_a_small_socp_through_the_c_abi() {
        let n = 3usize;
        let m = 5usize;
        // P = diag(2, 2, 0) (upper triangular CSC)
        let p_colptr = [0usize, 1, 2, 2];
        let p_rowval = [0usize, 1];
        let p_nzval = [2.0, 2.0];
        let q = [-6.0, -8.0, 1.0];
        // rows: [x0 + x1 = 1.5], [t <= 2], SOC [-t, -x0, -x1] + s = 0
        let a_colptr = [0usize, 2, 4, 6];
        let a_rowval = [0usize, 3, 0, 4, 1, 2];
        let a_nzval = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let b = [1.5, 2.0, 0.0, 0.0, 0.0];
        let socs = [3usize];
        let mut x = [0.0f64; 3];
        let mut diagnostics = HoverDiagnostics {
            status: 0,
            iterations: 0,
            objective: 0.0,
            solve_time_seconds: 0.0,
        };
        let status = unsafe {
            clarabel_conic_solve(
                n,
                m,
                p_colptr.as_ptr(),
                p_rowval.as_ptr(),
                p_nzval.as_ptr(),
                p_nzval.len(),
                q.as_ptr(),
                a_colptr.as_ptr(),
                a_rowval.as_ptr(),
                a_nzval.as_ptr(),
                a_nzval.len(),
                b.as_ptr(),
                1,
                1,
                socs.as_ptr(),
                socs.len(),
                100,
                x.as_mut_ptr(),
                x.len(),
                &mut diagnostics,
            )
        };
        assert_eq!(status, 1);
        assert!((x[0] - 0.3996).abs() < 1e-3, "{x:?}");
        assert!((x[1] - 1.1004).abs() < 1e-3, "{x:?}");
        assert!((x[2] - 1.1707).abs() < 1e-3, "{x:?}");
    }

    #[test]
    fn rejects_unsorted_or_lower_triangular_input() {
        let p_colptr = [0usize, 1];
        let p_rowval = [0usize];
        let p_nzval = [1.0];
        let lower = unsafe { read_csc(2, 1, [0usize, 1].as_ptr(), [1usize].as_ptr(), [1.0].as_ptr(), 1, true) };
        assert_eq!(lower.err(), Some(-13));
        let ok = unsafe { read_csc(1, 1, p_colptr.as_ptr(), p_rowval.as_ptr(), p_nzval.as_ptr(), 1, true) };
        assert!(ok.is_ok());
    }
}
