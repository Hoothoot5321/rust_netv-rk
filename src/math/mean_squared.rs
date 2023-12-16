pub fn mean_squared(prediction:f32,answer:f32) -> f32 {
    (answer-prediction).powf(2.0)
}
pub fn derived_mean_squared(prediction:f32,answer:f32)->f32 {
    2.0*(answer-prediction)
}
