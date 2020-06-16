use lin_reg::*;
use ndarray::prelude::*;

fn main() {
    
    let file_name = "example_data.csv"; 
    let matrix_raw = read_data(file_name);
    
    if let Ok(m) = matrix_raw {
        let y = m.slice(s![.., 0]);
        let x = m.slice(s![.., 1..]);
        let b_numerator = x.t().dot(&y);

        let xx = x.t().dot(&x);
        let b_denominator = invert_matrix(&xx);

        let beta = b_denominator.dot(&b_numerator);
        println!("result:\n {:?}", beta);

        let mut x_ = m.slice(s![.., 1..]).into_owned();
        rank(&mut x_);
    }
}

