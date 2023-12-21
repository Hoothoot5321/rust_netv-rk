pub fn mean_squared(prediction:f64,answer:f64) -> f64 {
    (answer-prediction).powf(2.0)
}
pub fn derived_mean_squared(prediction:f64,answer:f64)->f64 {
    2.0*(prediction-answer)
}
