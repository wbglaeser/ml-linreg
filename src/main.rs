use lin_reg::*;
use ndarray::prelude::*;

fn main() {
    
    let file_name = "example_data.csv"; 
    let matrix_raw = read_data(file_name);
    
    if let Ok(m) = matrix_raw {
        let y = m.slice(s![.., 0]).into_owned();
        let x = m.slice(s![.., 1..]).into_owned();
    
        let _b = linear_regression(&y, &x);
    }
}

