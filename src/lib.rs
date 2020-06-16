use std::error::Error;
//use csv::Reader;
use serde::Deserialize;

use ndarray_csv::Array2Reader;
use ndarray::prelude::*;
use ndarray::{Array, Array2, Ix2};
use csv::Reader;

#[derive(Debug, Deserialize)]
struct Record {
    y: f64,
    x0: f64,
    x1: f64,
    x2: f64,
}

// read in data from csv, currently very limited
pub fn read_data(file_name: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(file_name)?;
    //let mut matrix: Vec<Vec<f64>> = vec![];
    let matrix: Array2<f64> = rdr.deserialize_array2_dynamic()?;
    Ok(matrix)
}

// invert matrix using lu decomposition
pub fn invert_matrix(m: &Array2<f64>) -> Array<f64,Ix2> {
    
    // perform lu decomposition
    let lu_decomposition = lu_decomposition(m);
    let lower = lu_decomposition.0;
    let upper = lu_decomposition.1;
    
    // invert lus 
    let l_inverted = invert_lower(&lower);
    let u_inverted = invert_upper(&upper); 

    u_inverted.dot(&l_inverted)
}

// lu decomposition
pub fn lu_decomposition(m: &Array2<f64>) -> (Array<f64,Ix2>, Array<f64,Ix2>) {

    // retrieve n from shape
    let n = m.shape()[0];
    
    // initialise new matrizes
    let mut upper = Array::<f64,Ix2>::zeros((n, n).f());
    let mut lower = Array::<f64,Ix2>::zeros((n, n).f());

    // build compositions
    for i in 0..n {
        // start with upper
        for k in 0..n {

            let mut sum: f64 = 0.0;
            for j in 0..i {
                sum += lower[[i, j]] * upper[[j, k]]; 
            }

            upper[[i,k]] = m[[i, k]] - sum;
        }

        // do lower
        for k in 0..n {
            if i == k {
                lower[[i, i]] = 1.0;
            } else {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += lower[[k, j]] * upper[[j, i]];
                }
                lower[[k, i]] = (m[[k,i]] - sum) / upper[[i,i]];
            }
        }
    }
    (lower, upper)
}

// invert upper triagonal matrix
pub fn invert_upper(u: &Array2<f64>) -> Array::<f64,Ix2> {
    
    // setup empty inversion container
    let n = u.shape()[0];
    let mut u_invert = Array::<f64,Ix2>::zeros((n, n).f());

    // convert matrix
    for i in (0..n).rev() {
        for k in (0..i+1).rev() { 
            if i == k {
                u_invert[[i,i]] = 1. / u[[i,i]]
            } else {
                let mut sum = 0.0;
                for j in 0..n {
                    if k != j {
                        sum += u[[k, j]] * u_invert[[j,i]]               
                    } 
                }
                u_invert[[k,i]] = - sum / u[[k,k]]
            }
        }
    }
    u_invert
}

// invert lower triangular matrix
pub fn invert_lower(l: &Array2<f64>) -> Array::<f64,Ix2> {

    // setup empty inversion container
    let n = l.shape()[0];
    let mut l_invert = Array::<f64,Ix2>::zeros((n, n).f());

    // convert matrix
    for i in 0..n {
        for k in i..n { 
            if i == k {
                l_invert[[i,i]] = 1. / l[[i,i]]
            } else {
                let mut sum = 0.0;
                for j in 0..n {
                    if k != j {
                        sum += l[[k, j]] * l_invert[[j,i]]               
                    } 
                }
                l_invert[[k,i]] = - sum / l[[k,k]]
            }
        }
    }
    l_invert
}

fn gaussian_elemination(mat: &Array2<f64>) -> Array2<f64> {

    // retrieve shape of matrix
    let n = mat.shape()[0];
    let m = mat.shape()[1];

    let mut red_ef = mat.clone();

    for i in 0..m {
        
        if i < n {
            if red_ef[[i, i]] == 0. {
                let mut switch_row = 0;
                let mut max_val = 0.;
                for s in i..n {
                    if red_ef[[s, i]].abs() >= max_val {
                        switch_row = s;
                        max_val = red_ef[[s, i]].abs();
                    }
                }
                // switch rows
                let old_row = red_ef.slice(s![i, ..]).into_owned();
                let new_row = red_ef.slice(s![switch_row, ..]).into_owned();
                for (idx,v) in new_row.iter().enumerate() {
                    red_ef[[i, idx]] = *v; 
                }
                for (idx,v) in old_row.iter().enumerate() {
                    red_ef[[switch_row, idx]] = *v;
                }
            }
        
            // devide current row by first non-zero entry
            let divisor = red_ef[[i, i]].clone();
            for c in 0..m {
                red_ef[[i, c]] = red_ef[[i, c]] / divisor;
            }

            // subtract others rows
            for k in 0..n {    
                if k != i {        
                    let factor = -red_ef[[k, i]];
                    for j in 0..m {
                        red_ef[[k, j]] = red_ef[[k, j]] + factor * red_ef[[i, j]]
                    }
                }
            }
        }     
    }
    red_ef
}

// gaussian elimination algorithm to find rank
pub fn rank(mat: &mut Array2<f64>) {

    let reduced_ef = gaussian_elemination(&mat); 
    let n = reduced_ef.shape()[0];
    let m = reduced_ef.shape()[1];
    let mut rank = m.clone();

    for r in 0..n {
        if reduced_ef[[r,r]] != 1. {
            rank -= 1;
            continue;
        }

        for j in 0..r {
            if reduced_ef[[r,j]] != 0. {
                rank -= 1;
                continue;
            }
        }
        for c in 0..n {
            if c != r {
                if reduced_ef[[c,r]] != 0. {
                    rank -= 1;
                    continue;
                }
            }
        }


    } 
}

