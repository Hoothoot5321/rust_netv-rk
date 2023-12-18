pub fn relu(num:f64)->f64{
  let out = num*0.1; 
  if num>out {
          if num > 6.0 {
              6.0
          } 
          else {
              num
          }
  }
  else {
      out
  }
}
pub fn derived_relu(num:f64)-> f64{
    
  if num>0.0{
      1.0
  }
  else {
      0.1
  }
}
