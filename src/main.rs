

mod misc;
use misc::weight_funcs::{gen_layer};

mod classes;
use classes::{network::Network,neuron::Neuron};

mod math;
use math::{get_maks::get_ind_max};

use std::fs;

use csv::{Reader};


fn parse_answer(num:f32) -> Vec<f32> {
    const SIZE:usize = 10;
    let mut temp_vec = vec![0.0;SIZE];
    let temp = num as usize;
    temp_vec[temp] = 1.0; 
    temp_vec
} 
fn split_answer(reader:&mut Reader<&[u8]>) ->[Vec<Vec<f32>>;2]   {

    let mut answers:Vec<Vec<f32>> = vec![];
    let mut input :Vec<Vec<f32>> = vec![];
    for record in reader.records() {
        let record = record.unwrap();
        let mut temp_input:Vec<f32> = vec![]; 
        for (i,val) in record.iter().enumerate()  {
            let parsed:f32 = val.parse().unwrap();
            if i == 0 {
               let temp_answer:Vec<f32> = parse_answer(parsed);  
               answers.push(temp_answer);
            }
            else {
                temp_input.push((parsed/128.0)-1.0);
            }
        }
        input.push(temp_input);
    }
    return [answers,input]
}
fn load_test(reader:&mut Reader<&[u8]>) -> Vec<Vec<f32>> {

    let mut input :Vec<Vec<f32>> = vec![];
    for record in reader.records() {
        let record = record.unwrap();
        let mut temp_input:Vec<f32> = vec![]; 
        for (i,val) in record.iter().enumerate()  {
            let parsed:f32 = val.parse().unwrap();
            temp_input.push((parsed/128.0)-1.0);
        }
        input.push(temp_input);
    }
    input
} 



fn main() {
    let mut rng = rand::thread_rng();
    let load_weights = false;

    let path = r"C:\Users\MartinNammat\Documents\Programming-2\all_tests\rust_network_sigmoid\data\train.csv";

    let test_path = r"C:\Users\MartinNammat\Documents\Programming-2\all_tests\rust_network_sigmoid\data\test.csv";

    let weight_file= r"C:\Users\MartinNammat\Documents\Programming-2\all_tests\rust_network_sigmoid\weights.json";

    let file_content = fs::read_to_string(path).unwrap(); 
    let mut reader = Reader::from_reader(file_content.as_bytes());
    let [answers,input] = split_answer(&mut reader);
    /*
    let input = load_test(&mut reader);
    */

    let in_weights:Vec<Vec<Neuron>>;
   if load_weights == true {
    let weight_content= fs::read_to_string(weight_file).unwrap(); 
    let all_weights = serde_json::from_str::<Vec<Vec<Neuron>>>(&weight_content).unwrap(); 
    in_weights = all_weights;
   } 
   else {

    let layer_1 = gen_layer(15,784,&mut rng);
    let layer_output = gen_layer(10,15,&mut rng);

    let over_all = vec![layer_1,layer_output]; 
    in_weights = over_all;
   }
    
    let mut network = Network::new(in_weights,1.0); 
    network.train(answers,input,500000,&mut rng);
    /*
    for i in 0..100 {
    let output = network.feedforward(&input[i]);
    let pre_ind = get_ind_max(&output.out);
    println!("{}",pre_ind);
    }
    let ans_ind = get_ind_max(&answers[i]);
    println!("{}",ans_ind);
    println!("");
    }
    */
    
    





}

