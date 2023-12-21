pub fn cross_entropy_loss(prediction:f64,answer:f64) -> f64{
    if answer > 0.5 {
        prediction.ln()
    }
    else {
        (1.0-prediction).ln()
    }
}

pub fn derived_cross_entropy_loss(prediction:f64,answer:f64) -> f64{
    prediction-answer
}
