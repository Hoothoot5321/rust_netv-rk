use rand::{Rng};

mod misc;
use misc::weight_funcs::gen_weight;

mod classes;
use classes::{network::Network,neuron::Neuron};

mod math;

fn main() {
    let mut rng = rand::thread_rng();

    let weight1 = gen_weight(3,&mut rng); 
    let neuron1 = Neuron::new(weight1,rng.gen_range(0.0..1.0));

    let weight2 = gen_weight(3,&mut rng); 
    let neuron2 = Neuron::new(weight2,rng.gen_range(0.0..1.0));
    let layer1 = vec![neuron1,neuron2];

    let weight3 = gen_weight(2,&mut rng); 
    let neuron3 = Neuron::new(weight3,rng.gen_range(0.0..1.0));

    let weight4 = gen_weight(2,&mut rng); 
    let neuron4 = Neuron::new(weight4,rng.gen_range(0.0..1.0));
    let layer2 = vec![neuron3,neuron4];

    let overall = vec![layer1,layer2];


    let t0 = vec![1.0,0.0,1.0];
    let t1 = vec![0.0,1.0,0.0];
    let t2 = vec![1.0,1.0,1.0];
    let t3 = vec![0.0,0.0,1.0];
    let t4 = vec![1.0,1.0,0.0];
    let t5 = vec![0.0,1.0,1.0];

    let inputs = vec![t0,t1,t2,t3,t4,t5];
    
    let a0 = vec![1.0,0.0];
    let a1 = vec![0.0,1.0];
    let a2 = vec![0.0,1.0];
    let a3 = vec![0.0,1.0];
    let a4 = vec![1.0,0.0];
    let a5 = vec![1.0,0.0];




    let answers = vec![a0,a1,a2,a3,a4,a5];
    let mut network = Network::new(overall,0.01);
    network.train(answers,inputs,1000000,&mut rng);

    let temp = vec![0.0,1.0,1.0]; 
    let out = network.feedforward(&temp);
    println!("{:?}",out.out);


    let temp2 = vec![0.0,0.0,1.0]; 
    let out2 = network.feedforward(&temp2);
    println!("{:?}",out2.out);

    let temp3 = vec![1.0,1.0,1.0]; 
    let out3 = network.feedforward(&temp3);
    println!("{:?}",out3.out);
}

