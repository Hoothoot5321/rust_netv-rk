use rand::{Rng,rngs::ThreadRng};

#[allow(dead_code)]
pub fn gen_weight(amount:i32,rng: &mut ThreadRng ) -> Vec<f32> {
    let mut returner:Vec<f32> = vec![];
    for _ in 0..amount {
            let rand:f32 = rng.gen_range(0.0..1.0); 
            returner.push(rand);
    }
    return returner;
}
#[allow(dead_code)]
pub fn mult_weights(weight1:&Vec<f32>,weight2:&Vec<f32>) -> Vec<f32> {
    weight1.iter().zip(weight2).map(|(val1,val2)| val1*val2).collect::<Vec<f32>>()
}


#[allow(dead_code)]
pub fn update_weight(weight1:&Vec<f32>,weight2:&Vec<f32>) -> Vec<f32> {
    weight1.iter().zip(weight2).map(|(val1,val2)| val1-val2).collect::<Vec<f32>>()
}

#[allow(dead_code)]
pub fn dot_vecs(weight1:&Vec<f32>,weight2:&Vec<f32>) -> f32 {
    weight1.iter().zip(weight2).map(|(val1,val2)| val1*val2).fold(0.0,|acc,val|acc+val)
}
