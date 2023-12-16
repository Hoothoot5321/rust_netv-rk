
pub fn sigmoid(num:f32)->f32 {
    1.0/(1.0+(-num).exp())
}
pub fn derived_sigmoid(num:f32) ->f32 {
    let fx = sigmoid(num);
    fx*(1.0-fx)
}
