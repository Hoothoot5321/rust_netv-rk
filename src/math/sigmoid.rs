
pub fn sigmoid(num:f64)->f64 {
    1.0/(1.0+(-num).exp())
}
pub fn derived_sigmoid(num:f64) ->f64 {
    let fx = sigmoid(num);
    fx*(1.0-fx)
}
