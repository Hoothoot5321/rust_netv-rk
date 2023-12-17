pub fn cross_entropy(prediction:&Vec<f32>,answers:&Vec<f32>) ->f32  {
    let out = -(prediction.iter().zip(answers).fold(0.0,|acc,(pred,ans)|acc+ans*pred.ln()));
    out
}
