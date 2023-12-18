pub fn softmax(arr:&Vec<f64>) -> Vec<f64> {
    let soft_sum = arr.iter().fold(0.0,|acc,val|acc+val.exp());
    let temp_arr = arr.iter().map(|val|val.exp()/soft_sum).collect(); 
    
    temp_arr
}
