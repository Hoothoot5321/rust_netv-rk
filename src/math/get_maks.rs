
pub fn get_ind_max(arr:&Vec<f64>) -> usize {
    let mut pre_val = &(0.0);
    let mut index = 0;
    for (i,val) in arr.iter().enumerate() {
            if val > &pre_val {
                pre_val = val;
                index = i;
            }
    }
    index
}

pub fn get_val_max(arr:&Vec<f64>) -> f64{
    let mut pre_val = &(0.0);
    let mut index = 0;
    for (i,val) in arr.iter().enumerate() {
            if val > &pre_val {
                pre_val = val;
                index = i;
            }
    }
    *pre_val
}
